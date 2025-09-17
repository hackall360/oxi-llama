use std::collections::HashMap;

use api::{
    default_options, format_params, ChatRequest, DefValue, Duration, GenerateRequest, Message,
    PropertyType, ThinkValue, ToolFunction, ToolFunctionParameters, ToolProperty,
};
use serde_json::{self, json};

#[test]
fn keep_alive_parsing_from_json() {
    let cases = vec![
        ("{}", None),
        (
            "{ \"keep_alive\": 42 }",
            Some(Duration(std::time::Duration::from_secs(42))),
        ),
        (
            "{ \"keep_alive\": 42.5 }",
            Some(Duration(std::time::Duration::from_millis(42500))),
        ),
        (
            "{ \"keep_alive\": \"42m\" }",
            Some(Duration(std::time::Duration::from_secs(42 * 60))),
        ),
        (
            "{ \"keep_alive\": -1 }",
            Some(Duration(std::time::Duration::MAX)),
        ),
    ];
    for (req, exp) in cases {
        let dec: ChatRequest = serde_json::from_str(req).unwrap();
        assert_eq!(dec.keep_alive, exp);
    }
}

#[test]
fn duration_marshal_unmarshal() {
    let cases = vec![
        std::time::Duration::from_secs(42),
        std::time::Duration::from_secs(0),
    ];
    for d in cases {
        let b = serde_json::to_string(&Duration(d)).unwrap();
        let de: Duration = serde_json::from_str(&b).unwrap();
        assert_eq!(de.0, d);
    }
}

#[test]
fn use_mmap_parsing_from_json() {
    let tr = true;
    let fa = false;
    let cases = vec![
        ("{}", None),
        ("{ \"use_mmap\": true }", Some(tr)),
        ("{ \"use_mmap\": false }", Some(fa)),
    ];
    for (req, exp) in cases {
        let map: HashMap<String, serde_json::Value> = serde_json::from_str(req).unwrap();
        let mut opts = default_options();
        opts.from_map(&map).unwrap();
        assert_eq!(opts.use_mmap, exp.map(|b| b));
    }
}

#[test]
fn use_mmap_format_params() {
    let tr = true;
    let fa = false;
    let cases = vec![
        ("true", Some(tr), false),
        ("false", Some(fa), false),
        ("1", Some(tr), false),
        ("0", Some(fa), false),
        ("foo", None, true),
    ];
    for (val, exp, err) in cases {
        let mut m = HashMap::new();
        m.insert("use_mmap".to_string(), vec![val.to_string()]);
        let resp = format_params(m);
        if err {
            assert!(resp.is_err());
        } else {
            let map = resp.unwrap();
            assert_eq!(map.get("use_mmap").and_then(|v| v.as_bool()), exp);
        }
    }
}

#[test]
fn message_unmarshal_lowercase_role() {
    let tests = vec![
        ("{\"role\": \"USER\", \"content\": \"Hello!\"}", "user"),
        ("{\"role\": \"System\", \"content\": \"Init\"}", "system"),
    ];
    for (input, expected) in tests {
        let msg: Message = serde_json::from_str(input).unwrap();
        assert_eq!(msg.role, expected);
    }
}

#[test]
fn tool_function_unmarshal() {
    let input = "{\"name\":\"test\",\"description\":\"test function\",\"parameters\":{\"type\":\"object\",\"required\":[\"test\"],\"properties\":{\"test\":{\"type\":\"string\",\"description\":\"test prop\",\"enum\":[\"a\",\"b\"]}}}}";
    let tf: ToolFunction = serde_json::from_str(input).unwrap();
    assert_eq!(tf.name, "test");
}

#[test]
fn property_type_unmarshal_marshal() {
    let pt: PropertyType = serde_json::from_str("\"string\"").unwrap();
    assert_eq!(pt.0, vec!["string"]);
    let s = serde_json::to_string(&pt).unwrap();
    assert_eq!(s, "\"string\"");
    let pt2: PropertyType = serde_json::from_str("[\"string\",\"number\"]").unwrap();
    assert_eq!(pt2.0, vec!["string", "number"]);
}

#[test]
fn thinking_unmarshal() {
    let cases = vec![
        ("{ \"think\": true }", Some(ThinkValue::Bool(true))),
        (
            "{ \"think\": \"high\" }",
            Some(ThinkValue::Str("high".into())),
        ),
        ("{}", None),
    ];
    for (input, expected) in cases {
        let req: Result<GenerateRequest, _> = serde_json::from_str(input);
        if let Some(exp) = expected {
            let r = req.unwrap();
            assert_eq!(r.think, Some(exp));
        } else {
            assert!(req.unwrap().think.is_none());
        }
    }
}

#[test]
fn tool_function_parameters_string() {
    let params = ToolFunctionParameters {
        type_field: "object".into(),
        required: vec!["name".into()],
        properties: {
            let mut m = HashMap::new();
            m.insert(
                "name".into(),
                ToolProperty {
                    r#type: PropertyType(vec!["string".into()]),
                    description: "The name".into(),
                    ..Default::default()
                },
            );
            m
        },
        ..Default::default()
    };
    let expected = json!({"type":"object","required":["name"],"properties":{"name":{"type":"string","description":"The name"}}});
    let got: serde_json::Value = serde_json::from_str(&params.to_string()).unwrap();
    assert_eq!(got, expected);

    let params_fail = ToolFunctionParameters {
        type_field: "object".into(),
        defs: Some(DefValue::Fail),
        ..Default::default()
    };
    assert_eq!(params_fail.to_string(), "");
}

#[test]
fn tool_property_to_typescript_type() {
    let cases = vec![
        (
            ToolProperty {
                r#type: PropertyType(vec!["string".into()]),
                ..Default::default()
            },
            "string",
        ),
        (
            ToolProperty {
                r#type: PropertyType(vec!["number".into(), "null".into()]),
                ..Default::default()
            },
            "number | null",
        ),
        (
            ToolProperty {
                any_of: vec![
                    ToolProperty {
                        r#type: PropertyType(vec!["boolean".into()]),
                        ..Default::default()
                    },
                    ToolProperty {
                        r#type: PropertyType(vec!["object".into()]),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
            "boolean | Record<string, any>",
        ),
        (ToolProperty::default(), "any"),
    ];

    for (prop, expected) in cases {
        assert_eq!(prop.to_typescript_type(), expected);
    }
}
