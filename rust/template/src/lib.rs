use std::io::{self, Write};

use api::{Message, Tool, ToolProperty};
use go_template::{Context as GoContext, Template as GoTemplate};
use gtmpl_value::{Func, FuncError, Value as GoValue};
use include_dir::{include_dir, Dir};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Deserialize;
use serde_json::{self, json, Value as JsonValue};
use strsim::levenshtein;
use thiserror::Error;
use time::{macros::format_description, OffsetDateTime};

static TEMPLATE_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../../template");
static NAMED_TEMPLATES: Lazy<Vec<NamedTemplate>> =
    Lazy::new(|| load_templates().expect("embedded templates should be valid"));
static IDENT_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\.[A-Za-z_][A-Za-z0-9_]*").unwrap());

#[derive(Debug, Error)]
pub enum Error {
    #[error("template not found: {0}")]
    MissingTemplate(String),
    #[error("no matching template found")]
    NoMatchingTemplate,
    #[error(transparent)]
    Parse(#[from] go_template::error::ParseError),
    #[error(transparent)]
    Exec(#[from] go_template::error::ExecError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] io::Error),
}

#[derive(Clone, Deserialize, Default)]
pub struct TemplateParameters {
    #[serde(default)]
    pub stop: Vec<String>,
}

#[derive(Clone)]
pub struct NamedTemplate {
    pub name: String,
    pub template: String,
    bytes: Vec<u8>,
    pub parameters: Option<TemplateParameters>,
}

impl NamedTemplate {
    pub fn reader(&self) -> io::Cursor<Vec<u8>> {
        io::Cursor::new(self.bytes.clone())
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Default, Clone)]
pub struct Values {
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
    pub prompt: String,
    pub suffix: String,
    pub think: bool,
    pub think_level: String,
    pub is_think_set: bool,
    pub(crate) force_legacy: bool,
}

impl Values {
    pub fn force_legacy(mut self, force: bool) -> Self {
        self.force_legacy = force;
        self
    }
}

pub struct Template {
    inner: GoTemplate,
    raw: String,
}

impl Template {
    pub fn parse(source: &str) -> Result<Self, Error> {
        let mut template = new_template();
        template.parse(source)?;
        let raw = source.to_string();
        let mut result = Template {
            inner: template,
            raw,
        };
        let vars = result.vars();
        if !vars.iter().any(|v| v == "messages" || v == "response") {
            let mut augmented = new_template();
            let mut appended = source.to_string();
            appended.push_str("{{ .Response }}");
            augmented.parse(appended)?;
            result.inner = augmented;
        }
        Ok(result)
    }

    pub fn vars(&self) -> Vec<String> {
        let mut vars = IDENT_REGEX
            .find_iter(&self.raw)
            .filter_map(|m| {
                let ident = &self.raw[m.start() + 1..m.end()];
                if ident.is_empty() {
                    None
                } else {
                    Some(ident.to_lowercase())
                }
            })
            .collect::<Vec<_>>();
        vars.sort();
        vars.dedup();
        vars
    }

    pub fn contains(&self, needle: &str) -> bool {
        self.raw.contains(needle)
    }

    pub fn execute<W: Write>(&self, writer: &mut W, values: Values) -> Result<(), Error> {
        let (mut system_text, collated) = collate(&values.messages);

        if !values.prompt.is_empty() && !values.suffix.is_empty() {
            let rendered = self.render(json!({
                "Prompt": values.prompt,
                "Suffix": values.suffix,
                "Response": "",
                "Think": values.think,
                "ThinkLevel": values.think_level,
                "IsThinkSet": values.is_think_set,
            }))?;
            writer.write_all(rendered.as_bytes())?;
            return Ok(());
        }

        if !values.force_legacy && self.vars().iter().any(|v| v == "messages") {
            let rendered = self.render(json!({
                "System": system_text,
                "Messages": messages_to_json(&collated),
                "Tools": tools_to_json(&values.tools),
                "Response": "",
                "Think": values.think,
                "ThinkLevel": values.think_level,
                "IsThinkSet": values.is_think_set,
            }))?;
            writer.write_all(rendered.as_bytes())?;
            return Ok(());
        }

        let mut prompt = String::new();
        let mut response = String::new();
        let mut buffer = Vec::new();

        for message in collated {
            match message.role.as_str() {
                "system" => {
                    if !prompt.is_empty() || !response.is_empty() {
                        render_legacy(
                            &mut buffer,
                            self,
                            &system_text,
                            &prompt,
                            &response,
                            &values,
                        )?;
                        system_text.clear();
                        prompt.clear();
                        response.clear();
                    }
                    system_text = message.content.clone();
                }
                "user" => {
                    if !response.is_empty() {
                        render_legacy(
                            &mut buffer,
                            self,
                            &system_text,
                            &prompt,
                            &response,
                            &values,
                        )?;
                        system_text.clear();
                        prompt.clear();
                        response.clear();
                    }
                    prompt = message.content.clone();
                }
                "assistant" => {
                    response = message.content.clone();
                }
                _ => {}
            }
        }

        render_legacy(&mut buffer, self, &system_text, &prompt, "", &values)?;

        writer.write_all(&buffer)?;
        Ok(())
    }

    fn render(&self, value: JsonValue) -> Result<String, Error> {
        let context = GoContext::from(to_go_value(value));
        Ok(self.inner.render(&context)?)
    }
}

pub fn named(template: &str) -> Result<NamedTemplate, Error> {
    let mut best_score = usize::MAX;
    let mut best = None;
    for candidate in NAMED_TEMPLATES.iter() {
        let distance = levenshtein(template, &candidate.template);
        if distance < best_score {
            best_score = distance;
            best = Some(candidate.clone());
        }
    }

    if best_score < 100 {
        best.ok_or(Error::NoMatchingTemplate)
    } else {
        Err(Error::NoMatchingTemplate)
    }
}

fn render_legacy(
    buffer: &mut Vec<u8>,
    template: &Template,
    system: &str,
    prompt: &str,
    response: &str,
    values: &Values,
) -> Result<(), Error> {
    if system.is_empty() && prompt.is_empty() && response.is_empty() {
        return Ok(());
    }

    let rendered = template.render(json!({
        "System": system,
        "Prompt": prompt,
        "Response": response,
        "Think": values.think,
        "ThinkLevel": values.think_level,
        "IsThinkSet": values.is_think_set,
    }))?;
    buffer.extend_from_slice(rendered.as_bytes());
    Ok(())
}

fn collate(messages: &[Message]) -> (String, Vec<Message>) {
    let mut system = Vec::new();
    let mut collated: Vec<Message> = Vec::new();

    for message in messages {
        if message.role == "system" {
            system.push(message.content.clone());
        }

        if let Some(last) = collated.last_mut() {
            if last.role == message.role && message.role != "tool" {
                last.content.push_str("\n\n");
                last.content.push_str(&message.content);
                continue;
            }
        }

        collated.push(message.clone());
    }

    (system.join("\n\n"), collated)
}

fn messages_to_json(messages: &[Message]) -> JsonValue {
    JsonValue::Array(
        messages
            .iter()
            .map(|message| {
                json!({
                    "Role": message.role,
                    "Content": message.content,
                    "Thinking": message.thinking,
                    "Images": message.images,
                    "ToolCalls": message.tool_calls,
                    "ToolName": message.tool_name,
                })
            })
            .collect(),
    )
}

fn tools_to_json(tools: &[Tool]) -> JsonValue {
    JsonValue::Array(
        tools
            .iter()
            .map(|tool| {
                json!({
                    "Type": tool.type_field,
                    "Items": tool.items.clone(),
                    "Function": {
                        "Name": tool.function.name.clone(),
                        "Description": tool.function.description.clone(),
                        "Parameters": tool.function.parameters.clone(),
                    }
                })
            })
            .collect(),
    )
}

fn load_templates() -> Result<Vec<NamedTemplate>, Error> {
    let index = TEMPLATE_DIR
        .get_file("index.json")
        .ok_or_else(|| Error::MissingTemplate("index.json".into()))?;
    let entries: Vec<IndexEntry> = serde_json::from_slice(index.contents())?;

    let mut out = Vec::new();
    for entry in entries {
        let filename = format!("{}.gotmpl", entry.name);
        let file = TEMPLATE_DIR
            .get_file(&filename)
            .ok_or_else(|| Error::MissingTemplate(filename.clone()))?;
        let mut content = String::from_utf8_lossy(file.contents()).into_owned();
        if content.contains("\r\n") {
            content = content.replace("\r\n", "\n");
        }

        let params = TEMPLATE_DIR
            .get_file(&format!("{}.json", entry.name))
            .and_then(|f| serde_json::from_slice::<TemplateParameters>(f.contents()).ok());

        out.push(NamedTemplate {
            name: entry.name,
            template: entry.template,
            bytes: content.into_bytes(),
            parameters: params,
        });
    }

    Ok(out)
}

#[derive(Deserialize)]
struct IndexEntry {
    name: String,
    template: String,
}

fn new_template() -> GoTemplate {
    let mut template = GoTemplate::default();
    template.add_func("json", json_func as Func);
    template.add_func("currentDate", current_date_func as Func);
    template.add_func("toTypeScriptType", to_typescript_type_func as Func);
    template
}

fn json_func(args: &[GoValue]) -> Result<GoValue, FuncError> {
    if let Some(value) = args.first() {
        let json_value = from_go_value(value);
        let serialized = serde_json::to_string(&json_value)
            .map_err(|err| FuncError::Generic(err.to_string()))?;
        Ok(serialized.into())
    } else {
        Ok("null".into())
    }
}

fn current_date_func(_args: &[GoValue]) -> Result<GoValue, FuncError> {
    let now = OffsetDateTime::now_local().unwrap_or_else(|_| OffsetDateTime::now_utc());
    let format = format_description!("[year]-[month]-[day]");
    let formatted = now
        .format(&format)
        .map_err(|err| FuncError::Generic(err.to_string()))?;
    Ok(formatted.into())
}

fn to_typescript_type_func(args: &[GoValue]) -> Result<GoValue, FuncError> {
    if let Some(value) = args.first() {
        let json_value = from_go_value(value);
        let property: ToolProperty = serde_json::from_value(json_value)
            .map_err(|err| FuncError::Generic(err.to_string()))?;
        Ok(property.to_typescript_type().into())
    } else {
        Ok("any".into())
    }
}

fn to_go_value(value: JsonValue) -> GoValue {
    match value {
        JsonValue::Null => GoValue::Nil,
        JsonValue::Bool(b) => b.into(),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into()
            } else if let Some(u) = n.as_u64() {
                u.into()
            } else if let Some(f) = n.as_f64() {
                f.into()
            } else {
                GoValue::Nil
            }
        }
        JsonValue::String(s) => s.into(),
        JsonValue::Array(arr) => GoValue::Array(arr.into_iter().map(to_go_value).collect()),
        JsonValue::Object(map) => {
            GoValue::Map(map.into_iter().map(|(k, v)| (k, to_go_value(v))).collect())
        }
    }
}

fn from_go_value(value: &GoValue) -> JsonValue {
    match value {
        GoValue::NoValue | GoValue::Nil => JsonValue::Null,
        GoValue::Bool(b) => JsonValue::Bool(*b),
        GoValue::String(s) => JsonValue::String(s.clone()),
        GoValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                JsonValue::Number(i.into())
            } else if let Some(u) = n.as_u64() {
                JsonValue::Number(serde_json::Number::from(u))
            } else if let Some(f) = n.as_f64() {
                serde_json::Number::from_f64(f)
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Null)
            } else {
                JsonValue::Null
            }
        }
        GoValue::Array(arr) => JsonValue::Array(arr.iter().map(from_go_value).collect()),
        GoValue::Map(map) | GoValue::Object(map) => JsonValue::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), from_go_value(v)))
                .collect(),
        ),
        GoValue::Function(_) => JsonValue::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_vars() {
        let template =
            Template::parse("{{ .Prompt }} {{ .Response }}").expect("should parse template");
        assert!(template.vars().contains(&"prompt".into()));
        assert!(template.vars().contains(&"response".into()));
    }

    #[test]
    fn named_lookup() {
        let templates = super::load_templates().expect("templates load");
        let tpl = named(&templates[0].template);
        assert!(tpl.is_ok());
    }

    #[test]
    fn execute_messages_path() {
        let template = Template::parse("{{ range .Messages }}{{ .Role }} {{ .Content }}{{ end }}")
            .expect("parse");
        let mut output = Vec::new();
        let values = Values {
            messages: vec![
                Message {
                    role: "user".into(),
                    content: "hi".into(),
                    ..Default::default()
                },
                Message {
                    role: "assistant".into(),
                    content: "hello".into(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        template.execute(&mut output, values).expect("execute");
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("user hi"));
        assert!(rendered.contains("assistant hello"));
    }
}
