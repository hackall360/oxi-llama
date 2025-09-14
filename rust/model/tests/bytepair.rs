use model::{BytePairEncoding, TextProcessor};
use std::path::PathBuf;

fn load_llama_bpe() -> BytePairEncoding {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../../model/testdata/llama3.2");
    let encoder = path.join("encoder.json");
    let merges = path.join("vocab.bpe");
    BytePairEncoding::from_vocab_files(encoder.to_str().unwrap(), merges.to_str().unwrap()).unwrap()
}

#[test]
fn test_simple_encode_decode() {
    let tokenizer = load_llama_bpe();
    let ids = tokenizer.encode("hello world", true).unwrap();
    assert_eq!(ids, vec![15339, 1917]);
    let decoded = tokenizer.decode(&ids).unwrap();
    assert_eq!(decoded, "hello world");

    let ids = tokenizer.encode("hello <|end_of_text|>", true).unwrap();
    assert_eq!(ids, vec![15339, 220, 128001]);
}

#[test]
fn test_split() {
    let tokenizer = load_llama_bpe();
    let splits = tokenizer.split("Hello, WORLD!! How's it going?");
    assert_eq!(
        splits,
        vec!["Hello", ",", " WORLD", "!!", " How", "'s", " it", " going", "?",]
    );
}

#[test]
fn test_roundtrip_bytes() {
    let tokenizer = load_llama_bpe();
    for b in 0u8..=0xFF {
        let input = (b as char).to_string();
        let ids = tokenizer.encode(&input, false).unwrap();
        let output = tokenizer.decode(&ids).unwrap();
        assert_eq!(output, input);
    }
}
