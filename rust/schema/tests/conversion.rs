use schema::{json_schema_str_to_grammar, json_schema_to_grammar};
use serde_json::json;

#[test]
fn pattern_conversion_handles_repetition() {
    let schema = json!({
        "type": "string",
        "pattern": "^[a-z]{2}$"
    });
    let grammar = json_schema_to_grammar(&schema, true).expect("pattern conversion failed");
    assert!(
        grammar.contains("root-0{2,2}"),
        "grammar did not encode repetition: {grammar}"
    );
    assert!(grammar.contains("root"), "root rule missing: {grammar}");
}

#[test]
fn pattern_requires_anchors() {
    let schema = json!({
        "type": "string",
        "pattern": "[a-z]+"
    });
    let err =
        json_schema_to_grammar(&schema, true).expect_err("pattern without anchors should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("Pattern must start"),
        "unexpected error: {msg}"
    );
}

#[test]
fn object_with_additional_properties_generates_wildcard_rule() {
    let schema = json!({
        "type": "object",
        "properties": {
            "known": { "type": "string" }
        },
        "additionalProperties": true
    });
    let grammar = json_schema_to_grammar(&schema, true).expect("object conversion failed");
    assert!(grammar.contains("additional-kv"), "{grammar}");
    assert!(grammar.contains("known-kv"), "{grammar}");
}

#[test]
fn local_reference_is_resolved() {
    let schema = json!({
        "$defs": {
            "identifier": { "type": "string", "minLength": 1 }
        },
        "type": "object",
        "properties": {
            "id": { "$ref": "#/$defs/identifier" }
        },
        "required": ["id"],
        "additionalProperties": false
    });
    let grammar = json_schema_to_grammar(&schema, true).expect("ref conversion failed");
    assert!(
        grammar.contains("id ::= identifier"),
        "resolved rule missing: {grammar}"
    );
    assert!(grammar.contains("id-kv"), "{grammar}");
}

#[test]
fn integer_bounds_are_encoded() {
    let schema = json!({
        "type": "integer",
        "minimum": 2,
        "maximum": 5
    });
    let grammar = json_schema_to_grammar(&schema, true).expect("integer conversion failed");
    assert!(grammar.contains("[2-5]"), "{grammar}");
}

#[test]
fn json_string_input_round_trip() {
    let schema_str = r#"{ "type": "array", "items": { "type": "number" }, "minItems": 1 }"#;
    let grammar = json_schema_str_to_grammar(schema_str, true).expect("string conversion failed");
    assert!(grammar.contains("number"), "{grammar}");
}
