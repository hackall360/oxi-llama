use ollama_types::errtypes::{UnknownOllamaKey, UNKNOWN_OLLAMA_KEY_ERR_MSG};
use serde_json;

#[test]
fn unknown_key_display_and_serde() {
    let err = UnknownOllamaKey { key: "abc".into() };
    assert_eq!(
        err.to_string(),
        format!("unauthorized: {} \"abc\"", UNKNOWN_OLLAMA_KEY_ERR_MSG)
    );
    let json = serde_json::to_string(&err).unwrap();
    assert_eq!(json, "{\"key\":\"abc\"}");
    let de: UnknownOllamaKey = serde_json::from_str(&json).unwrap();
    assert_eq!(de.key, "abc");
}
