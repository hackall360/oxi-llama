use std::collections::HashMap;
use std::fmt;
use std::time::Duration as StdDuration;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use humantime::parse_duration;
use ollama_types::model::Capability;

pub type ImageData = Vec<u8>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusError {
    #[serde(skip)]
    pub status_code: u16,
    #[serde(default)]
    pub status: String,
    #[serde(default, rename = "error")]
    pub error_message: String,
}

impl fmt::Display for StatusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.status.is_empty() && !self.error_message.is_empty() {
            write!(f, "{}: {}", self.status, self.error_message)
        } else if !self.status.is_empty() {
            write!(f, "{}", self.status)
        } else if !self.error_message.is_empty() {
            write!(f, "{}", self.error_message)
        } else {
            write!(f, "something went wrong, please see the ollama server logs for details")
        }
    }
}

impl std::error::Error for StatusError {}

// ------------ Duration ------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Duration(pub StdDuration);

impl Serialize for Duration {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.0 == StdDuration::MAX {
            serializer.serialize_str("-1")
        } else {
            serializer.serialize_str(&humantime::format_duration(self.0).to_string())
        }
    }
}

impl<'de> Deserialize<'de> for Duration {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = Value::deserialize(deserializer)?;
        match v {
            Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    if f < 0.0 {
                        Ok(Duration(StdDuration::MAX))
                    } else {
                        Ok(Duration(StdDuration::from_secs_f64(f)))
                    }
                } else {
                    Err(serde::de::Error::custom("invalid number for duration"))
                }
            }
            Value::String(s) => {
                if s.starts_with('-') {
                    Ok(Duration(StdDuration::MAX))
                } else {
                    let d = parse_duration(&s).map_err(serde::de::Error::custom)?;
                    Ok(Duration(d))
                }
            }
            Value::Null => Ok(Duration(StdDuration::from_secs(0))),
            _ => Err(serde::de::Error::custom("unsupported type for duration")),
        }
    }
}

// ------------ ThinkValue ------------
#[derive(Debug, Clone, PartialEq)]
pub enum ThinkValue {
    Bool(bool),
    Str(String),
}

impl Serialize for ThinkValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            ThinkValue::Bool(b) => serializer.serialize_bool(*b),
            ThinkValue::Str(s) => serializer.serialize_str(s),
        }
    }
}

impl<'de> Deserialize<'de> for ThinkValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = Value::deserialize(deserializer)?;
        match v {
            Value::Bool(b) => Ok(ThinkValue::Bool(b)),
            Value::String(s) => {
                match s.as_str() {
                    "high" | "medium" | "low" => Ok(ThinkValue::Str(s)),
                    _ => Err(serde::de::Error::custom("invalid think value")),
                }
            }
            _ => Err(serde::de::Error::custom("think must be a boolean or string")),
        }
    }
}

// ------------ PropertyType ------------
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PropertyType(pub Vec<String>);

impl Serialize for PropertyType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.0.len() == 1 {
            serializer.serialize_str(&self.0[0])
        } else {
            self.0.serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for PropertyType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = Value::deserialize(deserializer)?;
        match v {
            Value::String(s) => Ok(PropertyType(vec![s])),
            Value::Array(arr) => {
                let mut out = Vec::new();
                for v in arr {
                    if let Value::String(s) = v {
                        out.push(s);
                    } else {
                        return Err(serde::de::Error::custom("invalid property type"));
                    }
                }
                Ok(PropertyType(out))
            }
            _ => Err(serde::de::Error::custom("invalid property type")),
        }
    }
}

// ------------ Tool related ------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    #[serde(default)]
    pub index: Option<i32>,
    pub name: String,
    pub arguments: ToolCallFunctionArguments,
}

pub type ToolCallFunctionArguments = HashMap<String, Value>;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Tool {
    #[serde(rename = "type")]
    pub type_field: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<Value>,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: ToolFunctionParameters,
}

impl ToString for ToolFunction {
    fn to_string(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

#[derive(Debug, Clone)]
pub enum DefValue {
    Value(Value),
    Fail,
}

impl Serialize for DefValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            DefValue::Value(v) => v.serialize(serializer),
            DefValue::Fail => Err(serde::ser::Error::custom("fail")),
        }
    }
}

impl<'de> Deserialize<'de> for DefValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = Value::deserialize(deserializer)?;
        Ok(DefValue::Value(v))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolFunctionParameters {
    #[serde(rename = "type")]
    pub type_field: String,
    #[serde(rename = "$defs", default, skip_serializing_if = "Option::is_none")]
    pub defs: Option<DefValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<Value>,
    #[serde(default)]
    pub required: Vec<String>,
    #[serde(default)]
    pub properties: HashMap<String, ToolProperty>,
}

impl ToolFunctionParameters {
    pub fn to_string(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolProperty {
    #[serde(rename = "anyOf", default, skip_serializing_if = "Vec::is_empty")]
    pub any_of: Vec<ToolProperty>,
    #[serde(rename = "type")]
    pub r#type: PropertyType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<Value>,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "enum", default, skip_serializing_if = "Vec::is_empty")]
    pub enum_values: Vec<Value>,
}

// ------------ Message ------------
#[derive(Debug, Clone, Serialize, Default)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub thinking: String,
    #[serde(default)]
    pub images: Vec<ImageData>,
    #[serde(default, rename = "tool_calls")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, rename = "tool_name")]
    pub tool_name: String,
}

impl<'de> Deserialize<'de> for Message {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Msg {
            #[serde(default)]
            role: String,
            content: String,
            #[serde(default)]
            thinking: String,
            #[serde(default)]
            images: Vec<ImageData>,
            #[serde(default, rename = "tool_calls")]
            tool_calls: Vec<ToolCall>,
            #[serde(default, rename = "tool_name")]
            tool_name: String,
        }
        let mut m = Msg::deserialize(deserializer)?;
        m.role = m.role.to_lowercase();
        Ok(Message {
            role: m.role,
            content: m.content,
            thinking: m.thinking,
            images: m.images,
            tool_calls: m.tool_calls,
            tool_name: m.tool_name,
        })
    }
}

// ------------ Requests ------------
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerateRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub suffix: String,
    #[serde(default)]
    pub system: String,
    #[serde(default)]
    pub template: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default)]
    pub raw: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<Duration>,
    #[serde(default)]
    pub images: Vec<ImageData>,
    #[serde(default)]
    pub options: HashMap<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub think: Option<ThinkValue>,
    #[serde(default, rename = "_debug_render_only")]
    pub debug_render_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<Duration>,
    #[serde(default)]
    pub tools: Vec<Tool>,
    #[serde(default)]
    pub options: HashMap<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub think: Option<ThinkValue>,
    #[serde(default, rename = "_debug_render_only")]
    pub debug_render_only: bool,
}

// ------------ Responses ------------
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerateResponse {
    #[serde(default)]
    pub model: String,
    #[serde(rename = "created_at", default)]
    pub created_at: String,
    pub response: String,
    #[serde(default)]
    pub thinking: String,
    pub done: bool,
    #[serde(rename = "done_reason", default)]
    pub done_reason: String,
    #[serde(default)]
    pub context: Vec<i32>,
    #[serde(default)]
    pub metrics: Metrics,
    #[serde(default, rename = "tool_calls")]
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatResponse {
    #[serde(default)]
    pub model: String,
    #[serde(rename = "created_at", default)]
    pub created_at: String,
    pub message: Message,
    #[serde(rename = "done_reason", default)]
    pub done_reason: String,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub metrics: Metrics,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DebugInfo {
    #[serde(rename = "rendered_template")]
    pub rendered_template: String,
    #[serde(rename = "image_count", default)]
    pub image_count: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DebugTemplateResponse {
    pub model: String,
    #[serde(rename = "created_at")]
    pub created_at: String,
    #[serde(rename = "_debug_info")]
    pub debug_info: DebugInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Metrics {
    #[serde(rename = "total_duration", default)]
    pub total_duration: Option<u64>,
    #[serde(rename = "load_duration", default)]
    pub load_duration: Option<u64>,
    #[serde(rename = "prompt_eval_count", default)]
    pub prompt_eval_count: Option<i32>,
    #[serde(rename = "prompt_eval_duration", default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(rename = "eval_count", default)]
    pub eval_count: Option<i32>,
    #[serde(rename = "eval_duration", default)]
    pub eval_duration: Option<u64>,
}

// ------------ Options ------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Options {
    #[serde(rename = "num_keep", default)]
    pub num_keep: i64,
    #[serde(rename = "seed", default)]
    pub seed: i64,
    #[serde(rename = "num_predict", default)]
    pub num_predict: i64,
    #[serde(rename = "top_k", default)]
    pub top_k: i64,
    #[serde(rename = "top_p", default)]
    pub top_p: f32,
    #[serde(rename = "min_p", default)]
    pub min_p: f32,
    #[serde(rename = "typical_p", default)]
    pub typical_p: f32,
    #[serde(rename = "repeat_last_n", default)]
    pub repeat_last_n: i64,
    #[serde(rename = "temperature", default)]
    pub temperature: f32,
    #[serde(rename = "repeat_penalty", default)]
    pub repeat_penalty: f32,
    #[serde(rename = "presence_penalty", default)]
    pub presence_penalty: f32,
    #[serde(rename = "frequency_penalty", default)]
    pub frequency_penalty: f32,
    #[serde(rename = "stop", default)]
    pub stop: Vec<String>,
    // runner options
    #[serde(rename = "num_ctx", default)]
    pub num_ctx: i64,
    #[serde(rename = "num_batch", default)]
    pub num_batch: i64,
    #[serde(rename = "num_gpu", default)]
    pub num_gpu: i64,
    #[serde(rename = "main_gpu", default)]
    pub main_gpu: i64,
    #[serde(rename = "use_mmap", default, skip_serializing_if = "Option::is_none")]
    pub use_mmap: Option<bool>,
    #[serde(rename = "num_thread", default)]
    pub num_thread: i64,
}

impl Default for Options {
    fn default() -> Self {
        default_options()
    }
}

pub fn default_options() -> Options {
    Options {
        num_keep: 4,
        seed: -1,
        num_predict: -1,
        top_k: 40,
        top_p: 0.9,
        min_p: 0.0,
        typical_p: 1.0,
        repeat_last_n: 64,
        temperature: 0.8,
        repeat_penalty: 1.1,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        stop: Vec::new(),
        num_ctx: 4096,
        num_batch: 512,
        num_gpu: -1,
        main_gpu: 0,
        use_mmap: None,
        num_thread: 0,
    }
}

impl Options {
    pub fn from_map(&mut self, m: &HashMap<String, Value>) -> Result<(), String> {
        for (k, v) in m {
            match k.as_str() {
                "use_mmap" => {
                    if let Some(b) = v.as_bool() {
                        self.use_mmap = Some(b);
                    } else {
                        return Err(format!("option \"{}\" must be of type boolean", k));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

pub fn format_params(params: HashMap<String, Vec<String>>) -> Result<HashMap<String, Value>, String> {
    let mut out = HashMap::new();
    for (k, vals) in params {
        match k.as_str() {
            "use_mmap" => {
                if vals.is_empty() {
                    return Err("invalid bool value []".to_string());
                }
                let v0 = vals[0].to_lowercase();
                let b = match v0.as_str() {
                    "true" | "1" => Ok(true),
                    "false" | "0" => Ok(false),
                    _ => Err(format!("invalid bool value {:?}", vals)),
                }?;
                out.insert(k, Value::Bool(b));
            }
            _ => return Err(format!("unknown parameter '{}'", k)),
        }
    }
    Ok(out)
}

// ------------ Other request/response structures ------------
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbedRequest {
    pub model: String,
    pub input: Value,
    #[serde(rename = "keep_alive", default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<Duration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
    #[serde(default)]
    pub dimensions: i32,
    #[serde(default)]
    pub options: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbedResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,
    #[serde(rename = "total_duration", default)]
    pub total_duration: Option<u64>,
    #[serde(rename = "load_duration", default)]
    pub load_duration: Option<u64>,
    #[serde(rename = "prompt_eval_count", default)]
    pub prompt_eval_count: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbeddingRequest {
    pub model: String,
    pub prompt: String,
    #[serde(rename = "keep_alive", default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<Duration>,
    #[serde(default)]
    pub options: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbeddingResponse {
    pub embedding: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CreateRequest {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default)]
    pub quantize: String,
    #[serde(default)]
    pub from: String,
    #[serde(default)]
    pub files: HashMap<String, String>,
    #[serde(default)]
    pub adapters: HashMap<String, String>,
    #[serde(default)]
    pub template: String,
    #[serde(default)]
    pub license: Option<Value>,
    #[serde(default)]
    pub system: String,
    #[serde(default)]
    pub parameters: HashMap<String, Value>,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub name: String,
    #[serde(default, rename = "quantization")]
    pub quantization: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeleteRequest {
    pub model: String,
    #[serde(default)]
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ShowRequest {
    pub model: String,
    #[serde(default)]
    pub system: String,
    #[serde(default)]
    pub template: String,
    #[serde(default)]
    pub verbose: bool,
    #[serde(default)]
    pub options: HashMap<String, Value>,
    #[serde(default)]
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelDetails {
    #[serde(rename = "parent_model", default)]
    pub parent_model: String,
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub families: Vec<String>,
    #[serde(rename = "parameter_size", default)]
    pub parameter_size: String,
    #[serde(rename = "quantization_level", default)]
    pub quantization_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Tensor {
    pub name: String,
    #[serde(rename = "type", default)]
    pub type_field: String,
    #[serde(default)]
    pub shape: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ShowResponse {
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub modelfile: String,
    #[serde(default)]
    pub parameters: String,
    #[serde(default)]
    pub template: String,
    #[serde(default)]
    pub system: String,
    #[serde(default)]
    pub details: ModelDetails,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(rename = "model_info", default)]
    pub model_info: HashMap<String, Value>,
    #[serde(rename = "projector_info", default)]
    pub projector_info: HashMap<String, Value>,
    #[serde(default)]
    pub tensors: Vec<Tensor>,
    #[serde(default)]
    pub capabilities: Vec<Capability>,
    #[serde(rename = "modified_at", default)]
    pub modified_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CopyRequest {
    pub source: String,
    pub destination: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PullRequest {
    pub model: String,
    #[serde(default)]
    pub insecure: bool,
    #[serde(default)]
    pub username: String,
    #[serde(default)]
    pub password: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default)]
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProgressResponse {
    pub status: String,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub total: i64,
    #[serde(default)]
    pub completed: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PushRequest {
    pub model: String,
    #[serde(default)]
    pub insecure: bool,
    #[serde(default)]
    pub username: String,
    #[serde(default)]
    pub password: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default)]
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ListModelResponse {
    pub name: String,
    pub model: String,
    #[serde(rename = "modified_at")]
    pub modified_at: String,
    pub size: i64,
    pub digest: String,
    #[serde(default)]
    pub details: ModelDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ListResponse {
    pub models: Vec<ListModelResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessModelResponse {
    pub name: String,
    pub model: String,
    pub size: i64,
    pub digest: String,
    #[serde(default)]
    pub details: ModelDetails,
    #[serde(rename = "expires_at")]
    pub expires_at: String,
    #[serde(rename = "size_vram")]
    pub size_vram: i64,
    #[serde(rename = "context_length")]
    pub context_length: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessResponse {
    pub models: Vec<ProcessModelResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenResponse {
    pub token: String,
}

