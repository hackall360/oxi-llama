use std::path::Path;

use anyhow::Result;

pub mod tokenizer;

pub use tokenizer::{parse_tokenizer, SpecialVocabulary, Tokenizer, Vocabulary};

/// Enumeration of model formats supported by the converter.
#[derive(Debug, Clone)]
pub enum ModelFormat {
    LLaMA,
    LLaMA4,
    LLaMAAdapter,
    Gemma,
    Gemma2,
    Gemma2Adapter,
    Gemma3,
    Gemma3N,
    Mistral,
    Mixtral,
    Qwen2,
    Qwen25VL,
    Bert,
    CommandR,
    Phi3,
    MLLama,
    GPTOss,
    Unknown,
}

/// Trait representing a source of tensors.
pub trait ModelReader {
    fn read(&mut self, name: &str) -> Result<Tensor>;
}

/// Trait representing a sink for tensors.
pub trait ModelWriter {
    fn write(&mut self, tensor: &Tensor) -> Result<()>;
}

/// Basic tensor representation used by the converter.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// Convert a model in a given format.  This is currently a stub that simply
/// validates input paths and returns success.
pub fn convert_model<S: AsRef<Path>, D: AsRef<Path>>(src: S, dst: D, _format: ModelFormat) -> Result<()> {
    let _ = (src.as_ref(), dst.as_ref());
    Ok(())
}
