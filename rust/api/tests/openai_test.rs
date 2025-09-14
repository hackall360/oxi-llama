use api::openai::*;
use api as api;
use serde_json::json;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;

const PREFIX: &str = "data:image/jpeg;base64,";
const IMAGE: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

#[test]
fn test_from_chat_request() {
    #[derive(Debug)]
    struct Case {
        body: String,
        req: Option<api::ChatRequest>,
        err: Option<ErrorResponse>,
    }

    let false_v = Some(false);
    let true_v = Some(true);

    let cases = vec![
        Case {
            body: r#"{"model":"test-model","messages":[{"role":"user","content":"Hello"}]}"#.to_string(),
            req: Some(api::ChatRequest {
                model: "test-model".into(),
                messages: vec![api::Message {
                    role: "user".into(),
                    content: "Hello".into(),
                    ..Default::default()
                }],
                stream: false_v,
                options: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("temperature".into(), json!(1.0));
                    m.insert("top_p".into(), json!(1.0));
                    m
                },
                ..Default::default()
            }),
            err: None,
        },
        Case {
            body: r#"{"model":"test-model","messages":[{"role":"user","content":"Hello"}],"stream":true,"max_tokens":999,"seed":123,"stop":["\n","stop"],"temperature":3.0,"frequency_penalty":4.0,"presence_penalty":5.0,"top_p":6.0,"response_format":{"type":"json_object"}}"#.to_string(),
            req: Some(api::ChatRequest {
                model: "test-model".into(),
                messages: vec![api::Message {
                    role: "user".into(),
                    content: "Hello".into(),
                    ..Default::default()
                }],
                stream: true_v,
                format: Some(json!("json")),
                options: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("num_predict".into(), json!(999));
                    m.insert("seed".into(), json!(123));
                    m.insert("stop".into(), json!(["\n","stop"]));
                    m.insert("temperature".into(), json!(3.0));
                    m.insert("frequency_penalty".into(), json!(4.0));
                    m.insert("presence_penalty".into(), json!(5.0));
                    m.insert("top_p".into(), json!(6.0));
                    m
                },
                ..Default::default()
            }),
            err: None,
        },
        Case {
            body: r#"{"model":"test-model","messages":[{"role":"user","content":"Hello"}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":999,"seed":123,"stop":["\n","stop"],"temperature":3.0,"frequency_penalty":4.0,"presence_penalty":5.0,"top_p":6.0,"response_format":{"type":"json_object"}}"#.to_string(),
            req: Some(api::ChatRequest {
                model: "test-model".into(),
                messages: vec![api::Message {
                    role: "user".into(),
                    content: "Hello".into(),
                    ..Default::default()
                }],
                stream: true_v,
                format: Some(json!("json")),
                options: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("num_predict".into(), json!(999));
                    m.insert("seed".into(), json!(123));
                    m.insert("stop".into(), json!(["\n","stop"]));
                    m.insert("temperature".into(), json!(3.0));
                    m.insert("frequency_penalty".into(), json!(4.0));
                    m.insert("presence_penalty".into(), json!(5.0));
                    m.insert("top_p".into(), json!(6.0));
                    m
                },
                ..Default::default()
            }),
            err: None,
        },
        Case {
            body: format!(
                r#"{{"model":"test-model","messages":[{{"role":"user","content":[{{"type":"text","text":"Hello"}},{{"type":"image_url","image_url":"{}{}"}}]}}]}}"#,
                PREFIX, IMAGE
            ),
            req: Some(api::ChatRequest {
                model: "test-model".into(),
                messages: vec![
                    api::Message {
                        role: "user".into(),
                        content: "Hello".into(),
                        ..Default::default()
                    },
                    api::Message {
                        role: "user".into(),
                        content: String::new(),
                        images: vec![BASE64.decode(IMAGE).unwrap()],
                        ..Default::default()
                    },
                ],
                stream: false_v,
                options: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("temperature".into(), json!(1.0));
                    m.insert("top_p".into(), json!(1.0));
                    m
                },
                ..Default::default()
            }),
            err: None,
        },
        Case {
            body: r#"{"model":"test-model","messages":[{"role":"user","content":"What's the weather?"},{"role":"assistant","tool_calls":[{"id":"id","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\"}"}}]}]}"#.to_string(),
            req: Some(api::ChatRequest {
                model: "test-model".into(),
                messages: vec![
                    api::Message {
                        role: "user".into(),
                        content: "What's the weather?".into(),
                        ..Default::default()
                    },
                    api::Message {
                        role: "assistant".into(),
                        tool_calls: vec![api::ToolCall {
                            function: api::ToolCallFunction {
                                index: None,
                                name: "get_weather".into(),
                                arguments: {
                                    let mut m = std::collections::HashMap::new();
                                    m.insert("location".into(), json!("Paris"));
                                    m
                                },
                            },
                        }],
                        ..Default::default()
                    },
                ],
                stream: false_v,
                options: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("temperature".into(), json!(1.0));
                    m.insert("top_p".into(), json!(1.0));
                    m
                },
                ..Default::default()
            }),
            err: None,
        },
        Case {
            body: r#"{"model":"test-model","messages":[{"role":"user","content":"What's the weather like in Paris Today?"},{"role":"assistant","content":"Let's see what the weather is like in Paris","tool_calls":[{"id":"id","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Paris, France\", \"format\": \"celsius\"}"}}]}]}"#.to_string(),
            req: Some(api::ChatRequest {
                model: "test-model".into(),
                messages: vec![
                    api::Message {
                        role: "user".into(),
                        content: "What's the weather like in Paris Today?".into(),
                        ..Default::default()
                    },
                    api::Message {
                        role: "assistant".into(),
                        content: "Let's see what the weather is like in Paris".into(),
                        tool_calls: vec![api::ToolCall {
                            function: api::ToolCallFunction {
                                index: None,
                                name: "get_current_weather".into(),
                                arguments: {
                                    let mut m = std::collections::HashMap::new();
                                    m.insert("location".into(), json!("Paris, France"));
                                    m.insert("format".into(), json!("celsius"));
                                    m
                                },
                            },
                        }],
                        ..Default::default()
                    },
                ],
                stream: false_v,
                options: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("temperature".into(), json!(1.0));
                    m.insert("top_p".into(), json!(1.0));
                    m
                },
                ..Default::default()
            }),
            err: None,
        },
        Case {
            body: r#"{"model":"test-model","messages":[{"role":"user","content":"What's the weather like in Paris Today?"},{"role":"assistant","content":"","tool_calls":[{"id":"id","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Paris, France\", \"format\": \"celsius\"}"}}]}]}"#.to_string(),
            req: Some(api::ChatRequest {
                model: "test-model".into(),
                messages: vec![
                    api::Message {
                        role: "user".into(),
                        content: "What's the weather like in Paris Today?".into(),
                        ..Default::default()
                    },
                    api::Message {
                        role: "assistant".into(),
                        content: String::new(),
                        tool_calls: vec![api::ToolCall {
                            function: api::ToolCallFunction {
                                index: None,
                                name: "get_current_weather".into(),
                                arguments: {
                                    let mut m = std::collections::HashMap::new();
                                    m.insert("location".into(), json!("Paris, France"));
                                    m.insert("format".into(), json!("celsius"));
                                    m
                                },
                            },
                        }],
                        ..Default::default()
                    },
                ],
                stream: false_v,
                options: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("temperature".into(), json!(1.0));
                    m.insert("top_p".into(), json!(1.0));
                    m
                },
                ..Default::default()
            }),
            err: None,
        },
        Case {
            body: r#"{"model":"test-model","messages":[{"role":"user","content":"What's the weather like in Paris Today?"},{"role":"assistant","reasoning":"Let's see what the weather is like in Paris","tool_calls":[{"id":"id","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Paris, France\", \"format\": \"celsius\"}"}}]}]}"#.to_string(),
            req: Some(api::ChatRequest {
                model: "test-model".into(),
                messages: vec![
                    api::Message {
                        role: "user".into(),
                        content: "What's the weather like in Paris Today?".into(),
                        ..Default::default()
                    },
                    api::Message {
                        role: "assistant".into(),
                        thinking: "Let's see what the weather is like in Paris".into(),
                        tool_calls: vec![api::ToolCall {
                            function: api::ToolCallFunction {
                                index: None,
                                name: "get_current_weather".into(),
                                arguments: {
                                    let mut m = std::collections::HashMap::new();
                                    m.insert("location".into(), json!("Paris, France"));
                                    m.insert("format".into(), json!("celsius"));
                                    m
                                },
                            },
                        }],
                        ..Default::default()
                    },
                ],
                stream: false_v,
                options: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("temperature".into(), json!(1.0));
                    m.insert("top_p".into(), json!(1.0));
                    m
                },
                ..Default::default()
            }),
            err: None,
        },
        Case {
            body: r#"{"model":"test-model","messages":[{"role":"user","content":"What's the weather like in Paris Today?"},{"role":"assistant","tool_calls":[{"id":"id_abc","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Paris, France\", \"format\": \"celsius\"}"}}]},{"role":"tool","tool_call_id":"id_abc","content":"The weather in Paris is 20 degrees Celsius"}]}"#.to_string(),
            req: Some(api::ChatRequest {
                model: "test-model".into(),
                messages: vec![
                    api::Message {
                        role: "user".into(),
                        content: "What's the weather like in Paris Today?".into(),
                        ..Default::default()
                    },
                    api::Message {
                        role: "assistant".into(),
                        tool_calls: vec![api::ToolCall {
                            function: api::ToolCallFunction {
                                index: None,
                                name: "get_current_weather".into(),
                                arguments: {
                                    let mut m = std::collections::HashMap::new();
                                    m.insert("location".into(), json!("Paris, France"));
                                    m.insert("format".into(), json!("celsius"));
                                    m
                                },
                            },
                        }],
                        ..Default::default()
                    },
                    api::Message {
                        role: "tool".into(),
                        content: "The weather in Paris is 20 degrees Celsius".into(),
                        tool_name: "get_current_weather".into(),
                        ..Default::default()
                    },
                ],
                stream: false_v,
                options: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("temperature".into(), json!(1.0));
                    m.insert("top_p".into(), json!(1.0));
                    m
                },
                ..Default::default()
            }),
            err: None,
        },
        Case {
            body: r#"{"model":"test-model","messages":[{"role":"user","content":"What's the weather like in Paris Today?"},{"role":"assistant","tool_calls":[{"id":"id","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Paris, France\", \"format\": \"celsius\"}"}}]},{"role":"tool","name":"get_current_weather","content":"The weather in Paris is 20 degrees Celsius"}]}"#.to_string(),
            req: Some(api::ChatRequest {
                model: "test-model".into(),
                messages: vec![
                    api::Message {
                        role: "user".into(),
                        content: "What's the weather like in Paris Today?".into(),
                        ..Default::default()
                    },
                    api::Message {
                        role: "assistant".into(),
                        tool_calls: vec![api::ToolCall {
                            function: api::ToolCallFunction {
                                index: None,
                                name: "get_current_weather".into(),
                                arguments: {
                                    let mut m = std::collections::HashMap::new();
                                    m.insert("location".into(), json!("Paris, France"));
                                    m.insert("format".into(), json!("celsius"));
                                    m
                                },
                            },
                        }],
                        ..Default::default()
                    },
                    api::Message {
                        role: "tool".into(),
                        content: "The weather in Paris is 20 degrees Celsius".into(),
                        tool_name: "get_current_weather".into(),
                        ..Default::default()
                    },
                ],
                stream: false_v,
                options: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("temperature".into(), json!(1.0));
                    m.insert("top_p".into(), json!(1.0));
                    m
                },
                ..Default::default()
            }),
            err: None,
        },
        Case {
            body: r#"{"model":"test-model","messages":[{"role":"user","content":"What's the weather like in Paris?"}],"stream":true,"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather","parameters":{"type":"object","required":["location"],"properties":{"location":{"type":"string","description":"The city and state"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}}}}}]}"#.to_string(),
            req: Some(api::ChatRequest {
                model: "test-model".into(),
                messages: vec![api::Message {
                    role: "user".into(),
                    content: "What's the weather like in Paris?".into(),
                    ..Default::default()
                }],
                tools: vec![api::Tool {
                    type_field: "function".into(),
                    items: None,
                    function: api::ToolFunction {
                        name: "get_weather".into(),
                        description: "Get the current weather".into(),
                        parameters: api::ToolFunctionParameters {
                            type_field: "object".into(),
                            defs: None,
                            items: None,
                            required: vec!["location".into()],
                            properties: {
                                let mut p = std::collections::HashMap::new();
                                p.insert(
                                    "location".into(),
                                    api::ToolProperty {
                                        any_of: vec![],
                                        r#type: api::PropertyType(vec!["string".into()]),
                                        items: None,
                                        description: "The city and state".into(),
                                        enum_values: vec![],
                                    },
                                );
                                p.insert(
                                    "unit".into(),
                                    api::ToolProperty {
                                        any_of: vec![],
                                        r#type: api::PropertyType(vec!["string".into()]),
                                        items: None,
                                        description: String::new(),
                                        enum_values: vec![json!("celsius"), json!("fahrenheit")],
                                    },
                                );
                                p
                            },
                        },
                    },
                }],
                stream: true_v,
                options: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("temperature".into(), json!(1.0));
                    m.insert("top_p".into(), json!(1.0));
                    m
                },
                ..Default::default()
            }),
            err: None,
        },
        Case {
            body: r#"{"model":"test-model","messages":[{"role":"user","content":2}]}"#.to_string(),
            req: None,
            err: Some(new_error("invalid message content type: number")),
        },
    ];

    for case in cases {
        let req: ChatCompletionRequest = serde_json::from_str(&case.body).unwrap();
        match from_chat_request(req) {
            Ok(actual) => {
                let expected = case.req.unwrap();
                assert_eq!(expected.model, actual.model, "model mismatch");
                assert_eq!(expected.messages, actual.messages);
            }
            Err(err) => {
                assert_eq!(case.err.unwrap(), err);
            }
        }
    }
}

#[test]
fn test_from_completion_request() {
    #[derive(Debug)]
    struct Case {
        body: &'static str,
        req: Option<api::GenerateRequest>,
        err: Option<ErrorResponse>,
    }

    let false_v = Some(false);
    let true_v = Some(true);

    let cases = vec![
        Case {
            body: r#"{"model":"test-model","prompt":"Hello","temperature":0.8,"stop":["\n","stop"],"suffix":"suffix"}"#,
            req: Some(api::GenerateRequest {
                model: "test-model".into(),
                prompt: "Hello".into(),
                suffix: "suffix".into(),
                stream: false_v,
                options: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("frequency_penalty".into(), json!(0.0));
                    m.insert("presence_penalty".into(), json!(0.0));
                    m.insert("temperature".into(), json!(0.8));
                    m.insert("top_p".into(), json!(1.0));
                    m.insert("stop".into(), json!(["\n","stop"]));
                    m
                },
                ..Default::default()
            }),
            err: None,
        },
        Case {
            body: r#"{"model":"test-model","prompt":"Hello","stream":true,"temperature":0.8,"stop":["\n","stop"],"suffix":"suffix"}"#,
            req: Some(api::GenerateRequest {
                model: "test-model".into(),
                prompt: "Hello".into(),
                suffix: "suffix".into(),
                stream: true_v,
                options: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("frequency_penalty".into(), json!(0.0));
                    m.insert("presence_penalty".into(), json!(0.0));
                    m.insert("temperature".into(), json!(0.8));
                    m.insert("top_p".into(), json!(1.0));
                    m.insert("stop".into(), json!(["\n","stop"]));
                    m
                },
                ..Default::default()
            }),
            err: None,
        },
        Case {
            body: r#"{"model":"test-model","prompt":"Hello","temperature":null,"stop":[1,2],"suffix":"suffix"}"#,
            req: None,
            err: Some(new_error("invalid type for 'stop' field")),
        },
    ];

    for case in cases {
        let req: CompletionRequest = serde_json::from_str(case.body).unwrap();
        match from_completion_request(req) {
            Ok(actual) => {
                let expected = case.req.unwrap();
                assert_eq!(expected.model, actual.model);
                assert_eq!(expected.prompt, actual.prompt);
                assert_eq!(expected.options, actual.options);
            }
            Err(err) => {
                assert_eq!(case.err.unwrap(), err);
            }
        }
    }
}

#[test]
fn test_from_embed_request() {
    let ok_body = r#"{"input":"Hello","model":"test-model"}"#;
    let req: EmbedRequest = serde_json::from_str(ok_body).unwrap();
    let out = from_embed_request(req).unwrap();
    assert_eq!(out.input, json!("Hello"));

    let err_body = r#"{"model":"test-model"}"#;
    let req: EmbedRequest = serde_json::from_str(err_body).unwrap();
    match from_embed_request(req) {
        Err(e) => assert_eq!(e, new_error("invalid input")),
        Ok(_) => panic!("expected error"),
    }
}

#[test]
fn test_to_list_and_model() {
    let list = api::ListResponse {
        models: vec![api::ListModelResponse {
            name: "test-model".into(),
            model: "test-model".into(),
            modified_at: "2023-06-16T00:03:22Z".into(),
            size: 0,
            digest: String::new(),
            details: Default::default(),
        }],
    };
    let lc = to_list_completion(list);
    assert_eq!(lc.object, "list");
    assert_eq!(lc.data.unwrap()[0].id, "test-model");

    let empty = api::ListResponse { models: vec![] };
    let lc_empty = to_list_completion(empty);
    assert_eq!(lc_empty.object, "list");
    assert!(lc_empty.data.is_none());

    let show = api::ShowResponse {
        modified_at: Some("2023-06-16T00:03:22Z".into()),
        ..Default::default()
    };
    let m = to_model(show, "test-model");
    assert_eq!(m.id, "test-model");
    assert_eq!(m.object, "model");
}

