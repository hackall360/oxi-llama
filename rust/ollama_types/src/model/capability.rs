use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Capability {
    Completion,
    Tools,
    Insert,
    Vision,
    Embedding,
    Thinking,
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Capability::Completion => "completion",
            Capability::Tools => "tools",
            Capability::Insert => "insert",
            Capability::Vision => "vision",
            Capability::Embedding => "embedding",
            Capability::Thinking => "thinking",
        };
        write!(f, "{}", s)
    }
}

impl FromStr for Capability {
    type Err = (); // simple error
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "completion" => Ok(Capability::Completion),
            "tools" => Ok(Capability::Tools),
            "insert" => Ok(Capability::Insert),
            "vision" => Ok(Capability::Vision),
            "embedding" => Ok(Capability::Embedding),
            "thinking" => Ok(Capability::Thinking),
            _ => Err(()),
        }
    }
}
