use fs::ggml::Tensor;
use fs::gguf::{write_gguf, GgufFile, Value};
use std::collections::HashMap;

#[test]
fn write_and_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");

    let mut kv = HashMap::new();
    kv.insert(
        "general.architecture".to_string(),
        Value::String("llama".into()),
    );
    kv.insert("llama.block_count".to_string(), Value::Uint32(1));
    kv.insert(
        "tokenizer.ggml.tokens".to_string(),
        Value::StringArray(vec!["hello".into(), "world".into()]),
    );
    kv.insert(
        "tokenizer.ggml.scores".to_string(),
        Value::Float32Array(vec![0.0, 1.0]),
    );

    let tensors = vec![
        Tensor::new("token_embd.weight", vec![2, 3], vec![0u8; 4 * 2 * 3]),
        Tensor::new("output.weight", vec![3, 2], vec![0u8; 4 * 3 * 2]),
    ];

    write_gguf(&path, &kv, &tensors).unwrap();

    let mut f = GgufFile::open(&path).unwrap();
    assert_eq!(
        f.key_value("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap(),
        "llama"
    );
    assert_eq!(f.num_key_values(), kv.len());
    assert_eq!(f.num_tensors(), tensors.len());
    let ti = f.tensor_info("token_embd.weight").unwrap();
    assert_eq!(ti.shape, vec![2, 3]);
    let (ti2, data) = f.tensor_reader("output.weight").unwrap();
    assert_eq!(ti2.shape, vec![3, 2]);
    assert_eq!(data.len() as u64, ti2.num_bytes());
}
