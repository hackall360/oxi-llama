use llama::{schema_to_grammar, schema_to_grammar_safe};

const ISSUE7978_JSON_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "steps": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "explanation": { "type": "string" },
          "output": { "type": "string" },
          "nested": {
            "type": "object",
            "properties": {
              "deep": { "type": "string" }
            }
          }
        },
        "required": ["explanation", "output"],
        "additionalProperties": false
      }
    },
    "final_answer": { "type": "string" },
    "01_numbered_key": { "type": "string" },
    "numbers": {
      "type": "array",
      "items": { "type": "number" }
    },
    "booleans": {
      "type": "array",
      "items": { "type": "boolean" }
    },
    "mixed": {
      "type": "array",
      "items": {
        "oneOf": [
          { "type": "string" },
          { "type": "number" },
          { "type": "boolean" }
        ]
      }
    }
  },
  "required": ["steps", "final_answer"],
  "additionalProperties": false
}"#;

#[test]
fn test_issue7978() {
    let g = schema_to_grammar_safe(ISSUE7978_JSON_SCHEMA)
        .expect("failed to convert JSON schema to grammar");
    let text = String::from_utf8(g).unwrap();

    let mut got = String::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("root ::=") {
            got = line.to_string();
            break;
        }
    }

    let want = r#"root ::= "{" space steps-kv "," space final-answer-kv ( "," space ( 01-numbered-key-kv 01-numbered-key-rest | numbers-kv numbers-rest | booleans-kv booleans-rest | mixed-kv ) )? "}" space"#;
    assert_eq!(got, want);
}

#[test]
fn test_schema_to_grammar() {
    struct Case {
        schema: &'static str,
        prefix: Option<&'static [u8]>,
    }
    let cases = [
        Case {
            schema: "invalid",
            prefix: None,
        },
        Case {
            schema: "{\"type\":\"object\"}",
            prefix: Some(b"root ::= object"),
        },
    ];
    for c in &cases {
        let safe = schema_to_grammar_safe(c.schema);
        let direct = schema_to_grammar(c.schema, true);
        match (safe, direct, c.prefix) {
            (None, Err(_), None) => {}
            (Some(buf), Ok(text), Some(prefix)) => {
                let buf_str = String::from_utf8_lossy(&buf);
                let prefix_str = String::from_utf8_lossy(prefix);
                assert!(
                    buf_str.contains(prefix_str.as_ref()),
                    "grammar = {:?}, want snippet {:?}",
                    buf_str,
                    prefix_str
                );
                assert!(
                    text.contains(prefix_str.as_ref()),
                    "text = {:?}, want snippet {:?}",
                    text,
                    prefix_str
                );
            }
            other => panic!("unexpected combination: {other:?}"),
        }
    }
}
