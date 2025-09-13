use model::{SentencePieceModel, TextProcessor, Vocabulary, TOKEN_TYPE_BYTE, TOKEN_TYPE_NORMAL};
use std::path::PathBuf;

fn load_sentencepiece_vocab() -> SentencePieceModel {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../../model/testdata/gemma2/tokenizer.model");
    let vocab = Vocabulary::new();
    SentencePieceModel::from_file(path.to_str().unwrap(), vocab).unwrap()
}

#[test]
fn test_sentencepiece_encode_roundtrip() {
    let tokenizer = load_sentencepiece_vocab();
    let cases: Vec<String> = vec![
        "hello".into(),
        "hello ".into(),
        "hello  ".into(),
        " hello".into(),
        " hello ".into(),
        " hello  ".into(),
        "hello world".into(),
        "请考试我的软件！12345".into(),
        "你好".into(),
        "Hello 你好 world!".into(),
        "Special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?".into(),
        "Multilingual: 你好 こんにちは Привет Hola مرحبا".into(),
        "Numbers and symbols: 123456789 +- */".into(),
        "Special tokens: <bos> text <eos>".into(),
        "Code snippets: func main() { fmt.Println(\"Hello World\") }".into(),
        format!(
            "Long text: {}{}{}",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ",
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
        ),
    ];
    for want in cases {
        let ids = tokenizer.encode(&want, true).unwrap();
        let got = tokenizer.decode(&ids).unwrap();
        assert_eq!(got, want, "ids {:?}", ids);
    }
}

#[test]
fn test_sentencepiece_special_tokens() {
    let tokenizer = load_sentencepiece_vocab();
    let cases = vec![("<bos>", vec![2]), ("<eos>", vec![1])];
    for (token, expected) in cases {
        let ids = tokenizer.encode(token, true).unwrap();
        assert_eq!(ids, expected);
    }
}

#[test]
fn test_decode_byte_tokens() {
    let mut vocab = Vocabulary::new();
    vocab.values = vec![
        "normal".into(),
        "<0xEA>".into(),
        "<0x41>".into(),
        "<0xC3>".into(),
        "<0xA3>".into(),
    ];
    vocab.types = vec![
        TOKEN_TYPE_NORMAL,
        TOKEN_TYPE_BYTE,
        TOKEN_TYPE_BYTE,
        TOKEN_TYPE_BYTE,
        TOKEN_TYPE_BYTE,
    ];
    vocab.scores = vec![0.0; 5];
    let spm = SentencePieceModel::new(vocab);
    let tests = vec![
        ("single byte token", vec![1], "ê".to_string()),
        ("ASCII byte token", vec![2], "A".to_string()),
        (
            "multiple byte tokens forming UTF-8 character",
            vec![3, 4],
            "ã".to_string(),
        ),
    ];
    for (name, ids, expected) in tests {
        let result = spm.decode(&ids).unwrap();
        assert_eq!(result, expected, "{}", name);
    }
}
