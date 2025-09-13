use serde::{Deserialize, Serialize};
use std::fmt;

pub const UNKNOWN_OLLAMA_KEY_ERR_MSG: &str = "unknown ollama key";
pub const INVALID_MODEL_NAME_ERR_MSG: &str = "invalid model name";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnknownOllamaKey {
    pub key: String,
}

impl fmt::Display for UnknownOllamaKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unauthorized: {} \"{}\"",
            UNKNOWN_OLLAMA_KEY_ERR_MSG,
            self.key.trim()
        )
    }
}

impl std::error::Error for UnknownOllamaKey {}
