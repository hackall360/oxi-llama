use std::path::Path;

use convert::{convert_model, ModelFormat};

#[test]
fn convert_stub_runs() {
    let src = Path::new(".");
    let dst = Path::new("model.bin");
    convert_model(src, dst, ModelFormat::LLaMA).unwrap();
}
