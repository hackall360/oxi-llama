use std::path::Path;

use convert::{convert_model, ModelFormat};
use fs::gguf::GgufFile;

#[test]
fn convert_writes_gguf() {
    let dir = tempfile::tempdir().unwrap();
    let dst = dir.path().join("out.gguf");
    convert_model(Path::new("."), &dst, ModelFormat::LLaMA).unwrap();
    GgufFile::open(&dst).unwrap();
}
