pub mod bytepair;
pub mod image_processor;
pub mod sentencepiece;
pub mod text_processor;
pub mod vocabulary;

pub use bytepair::BytePairEncoding;
pub use sentencepiece::SentencePieceModel;
pub use text_processor::TextProcessor;
pub use vocabulary::{
    Special, Vocabulary, TOKEN_TYPE_BYTE, TOKEN_TYPE_CONTROL, TOKEN_TYPE_NORMAL,
    TOKEN_TYPE_UNKNOWN, TOKEN_TYPE_UNUSED, TOKEN_TYPE_USER_DEFINED,
};

use fs::config::config_dir;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;

pub struct Tag {
    pub name: String,
    pub alternate: Vec<String>,
}

pub fn parse_tags(s: &str) -> Tag {
    let mut parts = s.split(',');
    let name = parts.next().unwrap_or("").to_string();
    let mut alt = Vec::new();
    for part in parts {
        if let Some(rest) = part.strip_prefix("alt:") {
            alt.push(rest.to_string());
        }
    }
    Tag {
        name,
        alternate: alt,
    }
}

/// Return the default directory used to load model configuration.
pub fn default_config_path() -> std::path::PathBuf {
    config_dir()
}

pub trait Model: Send + Sync {}

pub enum RegisteredModel {
    Text(Box<dyn TextProcessor>),
    Other(Box<dyn Model>),
}

pub type ModelConstructor = fn() -> RegisteredModel;

static MODELS: Lazy<Mutex<HashMap<String, ModelConstructor>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub fn register_model(name: &str, ctor: ModelConstructor) {
    let mut map = MODELS.lock().unwrap();
    if map.contains_key(name) {
        panic!("model already registered");
    }
    map.insert(name.to_string(), ctor);
}

#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    #[error("unsupported model architecture {0}")]
    Unsupported(String),
    #[error("not a TextProcessor")]
    NotTextProcessor,
}

pub fn get_text_processor(arch: &str) -> Result<Box<dyn TextProcessor>, ModelError> {
    let map = MODELS.lock().unwrap();
    let ctor = map
        .get(arch)
        .ok_or_else(|| ModelError::Unsupported(arch.to_string()))?;
    match ctor() {
        RegisteredModel::Text(tp) => Ok(tp),
        RegisteredModel::Other(_) => Err(ModelError::NotTextProcessor),
    }
}
