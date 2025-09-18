use llama::schema_to_grammar_safe;

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
        let g = schema_to_grammar_safe(c.schema);
        match (g, c.prefix) {
            (None, None) => {}
            (Some(buf), Some(prefix)) => {
                assert!(
                    buf.starts_with(prefix),
                    "grammar = {:?}, want prefix {:?}",
                    String::from_utf8_lossy(&buf),
                    String::from_utf8_lossy(prefix)
                );
            }
            (other, expected) => panic!(
                "unexpected combination: got {:?}, want {:?}",
                other.is_some(),
                expected.is_some()
            ),
        }
    }
}
