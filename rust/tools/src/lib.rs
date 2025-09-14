use std::collections::HashMap;
use serde_json::{Value, Map};
use api::{Tool, ToolCall, ToolCallFunction, ToolCallFunctionArguments};

#[derive(Clone, Copy, PartialEq)]
enum ToolsState {
    LookingForTag,
    ToolCalling,
    Done,
}

pub struct Parser {
    tag: String,
    tools: Vec<Tool>,
    state: ToolsState,
    buffer: Vec<u8>,
    n: i32,
}

impl Parser {
    pub fn new(template: &str, tools: Vec<Tool>) -> Self {
        let tag = parse_tag(template);
        Self::new_with_tag(tools, tag)
    }

    pub fn new_with_tag(tools: Vec<Tool>, tag: String) -> Self {
        Parser { tag, tools, state: ToolsState::LookingForTag, buffer: Vec::new(), n: 0 }
    }

    pub fn get_buffer(&self) -> &[u8] {
        &self.buffer
    }

    pub fn set_buffer(&mut self, data: &[u8]) {
        self.buffer = data.to_vec();
    }

    pub fn add(&mut self, s: &str) -> (Vec<ToolCall>, String) {
        if self.state == ToolsState::Done {
            return (Vec::new(), s.to_string());
        }

        self.buffer.extend_from_slice(s.as_bytes());

        let mut content = String::new();
        if self.state == ToolsState::LookingForTag {
            let (idx, found) = self.find_tag();
            if idx == -1 {
                content = String::from_utf8(self.buffer.clone()).unwrap_or_default();
                self.buffer.clear();
            } else {
                content = String::from_utf8(self.buffer[..idx as usize].to_vec()).unwrap_or_default();
                self.buffer.drain(..idx as usize);
            }

            if self.tag == "{" || self.tag == "[" {
                if content.trim().len() != 0 {
                    self.state = ToolsState::Done;
                    let mut rest = String::from_utf8(self.buffer.clone()).unwrap_or_default();
                    self.buffer.clear();
                    return (Vec::new(), content + &rest);
                }
            }

            if !found {
                return (Vec::new(), content);
            }

            self.state = ToolsState::ToolCalling;
        }

        let mut calls = Vec::new();
        loop {
            if let Some(call) = self.parse_tool_call() {
                calls.push(call);
            } else {
                break;
            }
        }

        if self.done() {
            self.state = ToolsState::Done;
            content = String::from_utf8(self.buffer.clone()).unwrap_or_default();
            self.buffer.clear();
        }

        (calls, content)
    }

    fn find_tag(&self) -> (isize, bool) {
        if let Some(i) = find_subslice(&self.buffer, self.tag.as_bytes()) {
            return (i as isize, true);
        }
        let max = std::cmp::min(self.buffer.len(), self.tag.len());
        for i in (1..=max).rev() {
            if self.buffer.ends_with(&self.tag.as_bytes()[..i]) {
                return ((self.buffer.len() - i) as isize, false);
            }
        }
        (-1, false)
    }

    fn parse_tool_call(&mut self) -> Option<ToolCall> {
        let (tool, mut end) = find_tool(&self.tools, &self.buffer)?;

        if let Some((args, i)) = find_arguments(&self.buffer) {
            if i > end { end = i; }
            let call = ToolCall { function: ToolCallFunction { index: Some(self.n), name: tool.function.name.clone(), arguments: args } };
            self.n += 1;
            self.buffer.drain(..end);
            Some(call)
        } else {
            None
        }
    }

    pub fn done(&self) -> bool {
        let (open, close) = match self.tag.as_str() {
            "{" => ('{', '}'),
            "[" => ('[', ']'),
            _ => return false,
        };
        let mut count = 0;
        for &c in &self.buffer {
            if c == open as u8 { count += 1; }
            else if c == close as u8 {
                count -= 1;
                if count == 0 { return true; }
            }
        }
        false
    }

    pub fn content(&self) -> String {
        if self.n > 0 { return String::new(); }
        if self.tag == "{" || self.tag == "[" { return String::from_utf8(self.buffer.clone()).unwrap_or_default(); }
        String::new()
    }
}

fn find_tool<'a>(tools: &'a [Tool], buf: &[u8]) -> Option<(&'a Tool, usize)> {
    if buf.is_empty() { return None; }
    let mut longest = "";
    for t in tools {
        if t.function.name.len() > longest.len() { longest = &t.function.name; }
    }
    for i in 1..=std::cmp::min(buf.len(), longest.len()) {
        let tail = &buf[buf.len()-i..];
        for t in tools {
            let name = t.function.name.as_bytes();
            if tail.len() < name.len() && name.starts_with(tail) {
                return None;
            }
        }
    }
    let mut found: Option<&Tool> = None;
    let mut start: isize = -1;
    let mut end: isize = -1;
    for t in tools {
        let name = t.function.name.as_bytes();
        if let Some(pos) = find_subslice(buf, name) {
            if start != -1 {
                if (pos as isize) > start { continue; }
                if (pos as isize) == start && name.len() <= found.unwrap().function.name.len() { continue; }
            }
            found = Some(t);
            start = pos as isize;
            end = (pos + name.len()) as isize;
        }
    }
    if let Some(f) = found { Some((f, end as usize)) } else { None }
}

fn find_arguments(buffer: &[u8]) -> Option<(ToolCallFunctionArguments, usize)> {
    if buffer.is_empty() { return None; }
    let mut start: isize = -1;
    let mut braces = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for i in 0..buffer.len() {
        let c = buffer[i];
        if escaped { escaped = false; continue; }
        if c == b'\\' { escaped = true; continue; }
        if c == b'"' { in_string = !in_string; continue; }
        if in_string { continue; }
        if c == b'{' {
            if braces == 0 { start = i as isize; }
            braces += 1;
        } else if c == b'}' {
            braces -= 1;
            if braces == 0 && start != -1 {
                let object = &buffer[start as usize ..= i];
                if let Ok(mut data) = serde_json::from_slice::<Map<String, Value>>(object) {
                    fn find_object(obj: &Map<String, Value>) -> Option<ToolCallFunctionArguments> {
                        if obj.contains_key("name") {
                            if let Some(Value::Object(args)) = obj.get("arguments") {
                                return Some(args.clone().into_iter().collect());
                            }
                            if let Some(Value::Object(args)) = obj.get("parameters") {
                                return Some(args.clone().into_iter().collect());
                            }
                            return Some(HashMap::new());
                        }
                        for v in obj.values() {
                            match v {
                                Value::Object(map) => {
                                    if let Some(res) = find_object(map) { return Some(res); }
                                }
                                Value::Array(arr) => {
                                    for item in arr {
                                        if let Value::Object(m) = item {
                                            if let Some(res) = find_object(m) { return Some(res); }
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        None
                    }
                    if let Some(args) = find_object(&data) {
                        return Some((args, i));
                    }
                    // return top-level object
                    let hm: HashMap<String, Value> = data.into_iter().collect();
                    return Some((hm, i));
                } else {
                    start = -1;
                    continue;
                }
            }
            if braces < 0 { braces = 0; }
        }
    }
    None
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

// ---- parse tag ----
pub fn parse_tag(tmpl: &str) -> String {
    if tmpl.is_empty() { return "{".to_string(); }
    let tmpl = tmpl.replace("\r\n", "\n");
    let mut pos = 0;
    while let Some(start) = tmpl[pos..].find("{{") {
        let start = pos + start;
        if let Some(end) = tmpl[start+2..].find("}}") {
            let end = start + 2 + end;
            let expr = &tmpl[start+2..end];
            if expr.contains(".ToolCalls") {
                let after = &tmpl[end+2..];
                let (tag, _) = scan_block(after);
                return tag.unwrap_or_else(|| "{".to_string());
            }
            pos = end + 2;
        } else { break; }
    }
    "{".to_string()
}

fn scan_block(s: &str) -> (Option<String>, usize) {
    let mut i = 0;
    while i < s.len() {
        if s[i..].starts_with("{{") {
            if let Some(end_off) = s[i+2..].find("}}") {
                let end = i + 2 + end_off;
                let action = s[i+2..end].trim();
                i = end + 2;
                if action.starts_with("if") || action.starts_with("range") || action.starts_with("with") {
                    let (text, consumed) = scan_block(&s[i..]);
                    i += consumed;
                    if text.is_some() { return (text, i); }
                } else if action.starts_with("else") {
                    let (text, consumed) = scan_block(&s[i..]);
                    i += consumed;
                    if text.is_some() { return (text, i); }
                } else if action.starts_with("end") {
                    return (None, i);
                } else {
                    continue;
                }
            } else {
                return (None, s.len());
            }
        } else {
            if let Some(next) = s[i..].find("{{") {
                let text = &s[i..i+next];
                if let Some(tag) = extract_tag(text) {
                    return (Some(tag), i+next);
                }
                i += next;
            } else {
                let text = &s[i..];
                if let Some(tag) = extract_tag(text) {
                    return (Some(tag), s.len());
                }
                return (None, s.len());
            }
        }
    }
    (None, i)
}

fn extract_tag(text: &str) -> Option<String> {
    if text.trim().is_empty() { return None; }
    let trimmed = text.trim_start();
    let cut = trimmed.find('{').unwrap_or(trimmed.len());
    let tag = trimmed[..cut].trim();
    if tag.is_empty() { None } else { Some(tag.to_string()) }
}

// Utility
