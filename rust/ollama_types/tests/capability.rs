use ollama_types::model::Capability;
use serde_json;

#[test]
fn capability_serialization() {
    let cap = Capability::Vision;
    let json = serde_json::to_string(&cap).unwrap();
    assert_eq!(json, "\"vision\"");
    let de: Capability = serde_json::from_str(&json).unwrap();
    assert_eq!(de, Capability::Vision);
}
