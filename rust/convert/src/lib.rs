use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use fs::gguf::{write_gguf, GgufFile, Value};

pub mod tokenizer;
pub mod tensor;
pub mod reader;

pub use tokenizer::{parse_tokenizer, SpecialVocabulary, Tokenizer, Vocabulary};
pub use tensor::{Tensor, TensorKind, Repacker, BaseTensor};
pub use reader::{ModelReader, ModelWriter, FsReader, VecWriter};

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

/// Convert a model in a given format. This is currently a stub that simply
/// validates input paths and returns success.
pub fn convert_model<S: AsRef<Path>, D: AsRef<Path>>(
    src: S,
    dst: D,
    _format: ModelFormat,
) -> Result<()> {
    if src.as_ref().is_file() {
        // Attempt to open the source as a GGUF file to exercise the reader. Any
        // error is ignored since conversion logic is not yet implemented.
        let _ = GgufFile::open(src.as_ref());
    }
    let mut kv = HashMap::new();
    kv.insert(
        "general.architecture".to_string(),
        Value::String("stub".into()),
    );
    write_gguf(dst.as_ref(), &kv, &[])?;
    Ok(())
}
