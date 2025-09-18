use std::collections::HashMap;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::OffsetDateTime;

use crate::types::{self as api, Message as ApiMessage};
use ollama_types::model::parse_name;

/// Error information returned to OpenAI compatible clients.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct Error {
    pub message: String,
    #[serde(rename = "type")]
    pub type_field: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ErrorResponse {
    pub error: Error,
}

/// Helper to create an `ErrorResponse` similar to the Go implementation.
pub fn new_error(message: &str) -> ErrorResponse {
    ErrorResponse {
        error: Error {
            message: message.to_string(),
            type_field: "invalid_request_error".to_string(),
            code: None,
            param: None,
        },
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct JsonSchema {
    pub schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ResponseFormat {
    #[serde(rename = "type", default)]
    pub type_field: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<JsonSchema>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Reasoning {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub type_field: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty", rename = "tool_calls")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "tool_call_id"
    )]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(
        default,
        rename = "frequency_penalty",
        skip_serializing_if = "Option::is_none"
    )]
    pub frequency_penalty: Option<f64>,
    #[serde(
        default,
        rename = "presence_penalty",
        skip_serializing_if = "Option::is_none"
    )]
    pub presence_penalty: Option<f64>,
    #[serde(default, rename = "top_p", skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub tools: Vec<api::Tool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    #[serde(
        default,
        rename = "reasoning_effort",
        skip_serializing_if = "Option::is_none"
    )]
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub frequency_penalty: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(default)]
    pub presence_penalty: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Value>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: f64,
    #[serde(default)]
    pub suffix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbedRequest {
    #[serde(default)]
    pub input: Value,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub created: i64,
    #[serde(rename = "owned_by")]
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ListCompletion {
    pub object: String,
    pub data: Option<Vec<Model>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: i32,
    #[serde(default)]
    pub completion_tokens: i32,
    #[serde(default)]
    pub total_tokens: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Choice {
    pub index: i32,
    pub message: Message,
    #[serde(rename = "finish_reason", skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ChunkChoice {
    pub index: i32,
    pub delta: Message,
    #[serde(rename = "finish_reason", skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct CompleteChunkChoice {
    pub text: String,
    pub index: i32,
    #[serde(rename = "finish_reason", skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ChatCompletion {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    #[serde(rename = "system_fingerprint")]
    pub system_fingerprint: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    #[serde(rename = "system_fingerprint")]
    pub system_fingerprint: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Completion {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    #[serde(rename = "system_fingerprint")]
    pub system_fingerprint: String,
    pub choices: Vec<CompleteChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    #[serde(rename = "system_fingerprint")]
    pub system_fingerprint: String,
    pub choices: Vec<CompleteChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Embedding {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct EmbeddingUsage {
    #[serde(default)]
    pub prompt_tokens: i32,
    #[serde(default)]
    pub total_tokens: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct EmbeddingList {
    pub object: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub data: Vec<Embedding>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<EmbeddingUsage>,
}

/// Convert chat completion request into an internal [`api::ChatRequest`].
pub fn from_chat_request(r: ChatCompletionRequest) -> Result<api::ChatRequest, ErrorResponse> {
    let mut messages: Vec<ApiMessage> = Vec::new();

    for msg in r.messages.iter() {
        let mut tool_name = String::new();
        if msg.role.eq_ignore_ascii_case("tool") {
            if let Some(n) = &msg.name {
                tool_name = n.clone();
            } else if let Some(id) = &msg.tool_call_id {
                tool_name = name_from_tool_call_id(&r.messages, id);
            }
        }

        match &msg.content {
            Value::String(s) => {
                let tool_calls = from_completion_tool_calls(&msg.tool_calls)?;
                messages.push(ApiMessage {
                    role: msg.role.clone(),
                    content: s.clone(),
                    thinking: msg.reasoning.clone().unwrap_or_default(),
                    images: Vec::new(),
                    tool_calls,
                    tool_name,
                });
            }
            Value::Array(arr) => {
                for c in arr {
                    let data = c
                        .as_object()
                        .ok_or_else(|| new_error("invalid message format"))?;
                    match data.get("type").and_then(|v| v.as_str()) {
                        Some("text") => {
                            let text = data
                                .get("text")
                                .and_then(|v| v.as_str())
                                .ok_or_else(|| new_error("invalid message format"))?;
                            messages.push(ApiMessage {
                                role: msg.role.clone(),
                                content: text.to_string(),
                                thinking: String::new(),
                                images: Vec::new(),
                                tool_calls: Vec::new(),
                                tool_name: String::new(),
                            });
                        }
                        Some("image_url") => {
                            let url_val = data
                                .get("image_url")
                                .ok_or_else(|| new_error("invalid message format"))?;
                            let url = if let Some(m) = url_val.as_object() {
                                m.get("url")
                                    .and_then(|v| v.as_str())
                                    .ok_or_else(|| new_error("invalid message format"))?
                            } else {
                                url_val
                                    .as_str()
                                    .ok_or_else(|| new_error("invalid message format"))?
                            };

                            let mut valid = false;
                            let mut b64 = String::new();
                            for t in ["jpeg", "jpg", "png", "webp"] {
                                let prefix = format!("data:image/{};base64,", t);
                                if let Some(rest) = url.strip_prefix(&prefix) {
                                    valid = true;
                                    b64 = rest.to_string();
                                    break;
                                }
                            }
                            if !valid {
                                return Err(new_error("invalid image input"));
                            }
                            let img = BASE64
                                .decode(b64.as_bytes())
                                .map_err(|_| new_error("invalid message format"))?;
                            messages.push(ApiMessage {
                                role: msg.role.clone(),
                                content: String::new(),
                                thinking: String::new(),
                                images: vec![img],
                                tool_calls: Vec::new(),
                                tool_name: String::new(),
                            });
                        }
                        _ => return Err(new_error("invalid message format")),
                    }
                }
                if !messages.is_empty() && !msg.tool_calls.is_empty() {
                    let tool_calls = from_completion_tool_calls(&msg.tool_calls)?;
                    if let Some(last) = messages.last_mut() {
                        last.tool_calls = tool_calls;
                        last.tool_name = tool_name;
                        if let Some(r) = &msg.reasoning {
                            last.thinking = r.clone();
                        }
                    }
                }
            }
            Value::Null => {
                if msg.tool_calls.is_empty() {
                    return Err(new_error(&format!(
                        "invalid message content type: {:?}",
                        msg.content
                    )));
                }
                let tool_calls = from_completion_tool_calls(&msg.tool_calls)?;
                messages.push(ApiMessage {
                    role: msg.role.clone(),
                    content: String::new(),
                    thinking: msg.reasoning.clone().unwrap_or_default(),
                    images: Vec::new(),
                    tool_calls,
                    tool_name,
                });
            }
            other => {
                if msg.tool_calls.is_empty() {
                    return Err(new_error(&format!(
                        "invalid message content type: {}",
                        value_type(other)
                    )));
                }
                let tool_calls = from_completion_tool_calls(&msg.tool_calls)?;
                messages.push(ApiMessage {
                    role: msg.role.clone(),
                    content: String::new(),
                    thinking: msg.reasoning.clone().unwrap_or_default(),
                    images: Vec::new(),
                    tool_calls,
                    tool_name,
                });
            }
        }
    }

    let mut options: HashMap<String, Value> = HashMap::new();
    if let Some(stop) = r.stop {
        match stop {
            Value::String(s) => {
                options.insert("stop".into(), Value::Array(vec![Value::String(s)]));
            }
            Value::Array(arr) => {
                let mut stops = Vec::new();
                for s in arr {
                    if let Value::String(st) = s {
                        stops.push(Value::String(st));
                    }
                }
                options.insert("stop".into(), Value::Array(stops));
            }
            _ => {}
        }
    }

    if let Some(m) = r.max_tokens {
        options.insert("num_predict".into(), Value::from(m));
    }
    options.insert(
        "temperature".into(),
        Value::from(r.temperature.unwrap_or(1.0)),
    );
    if let Some(seed) = r.seed {
        options.insert("seed".into(), Value::from(seed));
    }
    if let Some(fp) = r.frequency_penalty {
        options.insert("frequency_penalty".into(), Value::from(fp));
    }
    if let Some(pp) = r.presence_penalty {
        options.insert("presence_penalty".into(), Value::from(pp));
    }
    options.insert("top_p".into(), Value::from(r.top_p.unwrap_or(1.0)));

    let mut format: Option<Value> = None;
    if let Some(rf) = &r.response_format {
        match rf.type_field.trim().to_lowercase().as_str() {
            "json_object" => format = Some(Value::String("json".into())),
            "json_schema" => {
                if let Some(js) = &rf.json_schema {
                    format = Some(js.schema.clone());
                }
            }
            _ => {}
        }
    }

    let think = if let Some(rs) = &r.reasoning {
        rs.effort.clone().map(|v| api::ThinkValue::Str(v))
    } else if let Some(re) = &r.reasoning_effort {
        Some(api::ThinkValue::Str(re.clone()))
    } else {
        None
    };

    Ok(api::ChatRequest {
        model: r.model,
        messages,
        stream: Some(r.stream),
        format,
        keep_alive: None,
        tools: r.tools,
        options,
        think,
        debug_render_only: false,
    })
}

fn value_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn from_completion_tool_calls(
    tool_calls: &[ToolCall],
) -> Result<Vec<api::ToolCall>, ErrorResponse> {
    let mut out = Vec::new();
    for tc in tool_calls {
        let mut args: HashMap<String, Value> = HashMap::new();
        if !tc.function.arguments.is_empty() {
            args = serde_json::from_str(&tc.function.arguments)
                .map_err(|_| new_error("invalid tool call arguments"))?;
        }
        out.push(api::ToolCall {
            function: api::ToolCallFunction {
                index: tc.function.index,
                name: tc.function.name.clone(),
                arguments: args,
            },
        });
    }
    Ok(out)
}

fn name_from_tool_call_id(messages: &[Message], id: &str) -> String {
    for msg in messages.iter().rev() {
        for tc in &msg.tool_calls {
            if let Some(tc_id) = &tc.id {
                if tc_id == id {
                    return tc.function.name.clone();
                }
            }
        }
    }
    String::new()
}

/// Convert a completion request to [`api::GenerateRequest`].
pub fn from_completion_request(
    r: CompletionRequest,
) -> Result<api::GenerateRequest, ErrorResponse> {
    let mut options: HashMap<String, Value> = HashMap::new();

    if let Some(stop) = r.stop {
        match stop {
            Value::String(s) => {
                options.insert("stop".into(), Value::Array(vec![Value::String(s)]));
            }
            Value::Array(arr) => {
                let mut stops = Vec::new();
                for s in arr {
                    if let Value::String(st) = s {
                        stops.push(Value::String(st));
                    } else {
                        return Err(new_error("invalid type for 'stop' field"));
                    }
                }
                options.insert("stop".into(), Value::Array(stops));
            }
            other => {
                return Err(new_error(&format!(
                    "invalid type for 'stop' field: {}",
                    value_type(&other)
                )))
            }
        }
    }

    if let Some(m) = r.max_tokens {
        options.insert("num_predict".into(), Value::from(m));
    }

    options.insert(
        "temperature".into(),
        Value::from(r.temperature.unwrap_or(1.0)),
    );

    if let Some(seed) = r.seed {
        options.insert("seed".into(), Value::from(seed));
    }

    options.insert("frequency_penalty".into(), Value::from(r.frequency_penalty));
    options.insert("presence_penalty".into(), Value::from(r.presence_penalty));
    options.insert(
        "top_p".into(),
        Value::from(if r.top_p != 0.0 { r.top_p } else { 1.0 }),
    );

    Ok(api::GenerateRequest {
        model: r.model,
        prompt: r.prompt,
        suffix: r.suffix,
        stream: Some(r.stream),
        options,
        ..Default::default()
    })
}

/// Validate and convert an embedding creation request.
pub fn from_embed_request(r: EmbedRequest) -> Result<api::EmbedRequest, ErrorResponse> {
    if r.input.is_null() {
        return Err(new_error("invalid input"));
    }
    Ok(api::EmbedRequest {
        model: r.model,
        input: r.input,
        dimensions: r.dimensions.unwrap_or_default(),
        keep_alive: None,
        truncate: None,
        options: HashMap::new(),
    })
}

/// Convert a [`api::ListResponse`] into an OpenAI compatible model list.
pub fn to_list_completion(r: api::ListResponse) -> ListCompletion {
    let mut data = Vec::new();
    for m in r.models {
        let created = OffsetDateTime::parse(
            &m.modified_at,
            &time::format_description::well_known::Rfc3339,
        )
        .map(|t| t.unix_timestamp())
        .unwrap_or(0);
        let name = parse_name(&m.name);
        data.push(Model {
            id: m.name,
            object: "model".into(),
            created,
            owned_by: name.namespace,
        });
    }

    ListCompletion {
        object: "list".into(),
        data: if data.is_empty() { None } else { Some(data) },
    }
}

/// Convert a [`api::ShowResponse`] into an OpenAI model description.
pub fn to_model(r: api::ShowResponse, model: &str) -> Model {
    let created = r
        .modified_at
        .as_ref()
        .and_then(|s| {
            OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339)
                .ok()
                .map(|t| t.unix_timestamp())
        })
        .unwrap_or(0);
    let name = parse_name(model);
    Model {
        id: model.to_string(),
        object: "model".into(),
        created,
        owned_by: name.namespace,
    }
}

fn to_usage_chat(r: &api::ChatResponse) -> Usage {
    let prompt = r.metrics.prompt_eval_count.unwrap_or(0);
    let completion = r.metrics.eval_count.unwrap_or(0);
    Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
    }
}

fn tool_call_id() -> String {
    use rand::{distributions::Alphanumeric, Rng};
    let s: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(8)
        .map(char::from)
        .collect();
    format!("call_{}", s.to_lowercase())
}

fn to_tool_calls(tc: &[api::ToolCall]) -> Vec<ToolCall> {
    tc.iter()
        .map(|t| {
            let args = serde_json::to_string(&t.function.arguments).unwrap_or_default();
            ToolCall {
                id: Some(tool_call_id()),
                type_field: "function".into(),
                function: ToolCallFunction {
                    name: t.function.name.clone(),
                    arguments: args,
                    index: t.function.index,
                },
            }
        })
        .collect()
}

/// Convert an [`api::ChatResponse`] into an OpenAI ChatCompletion.
pub fn to_chat_completion(id: &str, r: api::ChatResponse) -> ChatCompletion {
    let usage = to_usage_chat(&r);
    let created = OffsetDateTime::parse(
        &r.created_at,
        &time::format_description::well_known::Rfc3339,
    )
    .map(|t| t.unix_timestamp())
    .unwrap_or(0);
    let model = r.model.clone();
    let done_reason = r.done_reason.clone();
    let message = r.message;
    let tool_calls = to_tool_calls(&message.tool_calls);
    let finish_reason = if !tool_calls.is_empty() {
        Some("tool_calls".into())
    } else if !done_reason.is_empty() {
        Some(done_reason)
    } else {
        None
    };

    ChatCompletion {
        id: id.to_string(),
        object: "chat.completion".into(),
        created,
        model,
        system_fingerprint: "fp_ollama".into(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: message.role,
                content: Value::String(message.content),
                reasoning: if message.thinking.is_empty() {
                    None
                } else {
                    Some(message.thinking)
                },
                tool_calls,
                name: None,
                tool_call_id: None,
            },
            finish_reason,
        }],
        usage: Some(usage),
    }
}

/// Convert a chat response chunk for streaming.
pub fn to_chat_chunk(id: &str, r: api::ChatResponse, tool_call_sent: bool) -> ChatCompletionChunk {
    let tool_calls = to_tool_calls(&r.message.tool_calls);
    let finish_reason = if !r.done_reason.is_empty() {
        if tool_call_sent || !tool_calls.is_empty() {
            Some("tool_calls".into())
        } else {
            Some(r.done_reason.clone())
        }
    } else {
        None
    };

    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".into(),
        created: OffsetDateTime::now_utc().unix_timestamp(),
        model: r.model,
        system_fingerprint: "fp_ollama".into(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Message {
                role: "assistant".into(),
                content: Value::String(r.message.content),
                reasoning: if r.message.thinking.is_empty() {
                    None
                } else {
                    Some(r.message.thinking)
                },
                tool_calls,
                name: None,
                tool_call_id: None,
            },
            finish_reason,
        }],
        usage: None,
    }
}

fn to_usage_generate(r: &api::GenerateResponse) -> Usage {
    let prompt = r.metrics.prompt_eval_count.unwrap_or(0);
    let completion = r.metrics.eval_count.unwrap_or(0);
    Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
    }
}

/// Convert a [`api::GenerateResponse`] into an OpenAI completion.
pub fn to_completion(id: &str, r: api::GenerateResponse) -> Completion {
    let usage = to_usage_generate(&r);
    let created = OffsetDateTime::parse(
        &r.created_at,
        &time::format_description::well_known::Rfc3339,
    )
    .map(|t| t.unix_timestamp())
    .unwrap_or(0);
    let model = r.model.clone();
    let text = r.response.clone();
    let finish_reason = if r.done_reason.is_empty() {
        None
    } else {
        Some(r.done_reason.clone())
    };
    Completion {
        id: id.to_string(),
        object: "text_completion".into(),
        created,
        model,
        system_fingerprint: "fp_ollama".into(),
        choices: vec![CompleteChunkChoice {
            text,
            index: 0,
            finish_reason,
        }],
        usage: Some(usage),
    }
}

/// Convert a streaming generate response chunk.
pub fn to_completion_chunk(id: &str, r: api::GenerateResponse) -> CompletionChunk {
    let finish_reason = if r.done_reason.is_empty() {
        None
    } else {
        Some(r.done_reason.clone())
    };
    CompletionChunk {
        id: id.to_string(),
        object: "text_completion".into(),
        created: OffsetDateTime::now_utc().unix_timestamp(),
        model: r.model,
        system_fingerprint: "fp_ollama".into(),
        choices: vec![CompleteChunkChoice {
            text: r.response,
            index: 0,
            finish_reason,
        }],
        usage: None,
    }
}

/// Convert an embedding response into an OpenAI embedding list.
pub fn to_embedding_list(model: &str, r: api::EmbedResponse) -> EmbeddingList {
    if r.embeddings.is_empty() {
        return EmbeddingList::default();
    }

    let data = r
        .embeddings
        .into_iter()
        .enumerate()
        .map(|(i, e)| Embedding {
            object: "embedding".into(),
            embedding: e,
            index: i as i32,
        })
        .collect();

    EmbeddingList {
        object: "list".into(),
        data,
        model: model.into(),
        usage: Some(EmbeddingUsage {
            prompt_tokens: r.prompt_eval_count.unwrap_or(0),
            total_tokens: r.prompt_eval_count.unwrap_or(0),
        }),
    }
}
