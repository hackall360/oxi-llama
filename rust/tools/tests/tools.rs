use std::collections::HashMap;
use serde_json::json;
use api::{Tool, ToolFunction, ToolFunctionParameters, ToolCall, ToolCallFunction};
use tools::Parser;

fn tools_list() -> Vec<Tool> {
    fn params() -> ToolFunctionParameters {
        ToolFunctionParameters { type_field: "object".to_string(), defs: None, items: None, required: vec![], properties: HashMap::new() }
    }
    vec![
        Tool { type_field: "function".to_string(), items: None, function: ToolFunction { name: "get_temperature".to_string(), description: String::new(), parameters: params() } },
        Tool { type_field: "function".to_string(), items: None, function: ToolFunction { name: "get_conditions".to_string(), description: String::new(), parameters: params() } },
        Tool { type_field: "function".to_string(), items: None, function: ToolFunction { name: "say_hello".to_string(), description: String::new(), parameters: params() } },
        Tool { type_field: "function".to_string(), items: None, function: ToolFunction { name: "say_hello_world".to_string(), description: String::new(), parameters: params() } },
        Tool { type_field: "function".to_string(), items: None, function: ToolFunction { name: "get_address".to_string(), description: String::new(), parameters: params() } },
        Tool { type_field: "function".to_string(), items: None, function: ToolFunction { name: "add".to_string(), description: String::new(), parameters: params() } },
    ]
}

#[test]
fn parser_basic_cases() {
    let tools = tools_list();
    let qwen = "{{if .ToolCalls}}<tool_call>{{range .ToolCalls}}{\"name\": \"{{.Function.Name}}\",\"arguments\": {{.Function.Arguments}}}{{end}}</tool_call>{{end}}";
    let list_tmpl = "{{if .ToolCalls}}[{{range .ToolCalls}}{\"name\": \"{{.Function.Name}}\", \"arguments\": {{.Function.Arguments}}}{{end}}]{{end}}";

    struct Case<'a> {
        name: &'a str,
        inputs: Vec<&'a str>,
        tmpl: &'a str,
        content: &'a str,
        calls: Vec<ToolCall>,
    }

    let cases = vec![
        Case { name: "no tool calls", inputs: vec!["Hello"], tmpl: qwen, content: "Hello", calls: vec![] },
        Case { name: "empty input", inputs: vec![""], tmpl: qwen, content: "", calls: vec![] },
        Case { name: "tool call", inputs: vec!["<tool_call>{\"name\": \"get_conditions\", \"arguments\": {\"location\": \"San Francisco\"}}</tool_call>"], tmpl: qwen, content: "", calls: vec![ToolCall { function: ToolCallFunction { index: Some(0), name: "get_conditions".to_string(), arguments: { let mut m=HashMap::new(); m.insert("location".to_string(), json!("San Francisco")); m } } }] },
        Case { name: "empty args", inputs: vec!["<tool_call>{\"name\": \"get_conditions\", \"arguments\": {}}</tool_call>"], tmpl: qwen, content: "", calls: vec![ToolCall { function: ToolCallFunction { index: Some(0), name: "get_conditions".to_string(), arguments: HashMap::new() } }] },
        Case { name: "text before tool call", inputs: vec!["Let me check <tool_call>{\"name\": \"get_temperature\", \"arguments\": {\"city\": \"New York\"}}</tool_call>"], tmpl: qwen, content: "Let me check ", calls: vec![ToolCall { function: ToolCallFunction { index: Some(0), name: "get_temperature".to_string(), arguments: { let mut m=HashMap::new(); m.insert("city".to_string(), json!("New York")); m } } }] },
        Case { name: "two tool calls in list", inputs: vec!["[TOOL_CALLS] [{\"name\": \"get_temperature\", \"arguments\": {\"city\": \"London\", \"format\": \"fahrenheit\"}}, {\"name\": \"get_conditions\", \"arguments\": {\"location\": \"Tokyo\"}}][/TOOL_CALLS]"], tmpl: "{{if .ToolCalls}}[TOOL_CALLS] [{{range .ToolCalls}}{\"name\": \"{{.Function.Name}}\", \"arguments\": {{.Function.Arguments}}}{{end}}][/TOOL_CALLS]{{end}}", content: "", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "get_temperature".to_string(), arguments: { let mut m=HashMap::new(); m.insert("city".to_string(), json!("London")); m.insert("format".to_string(), json!("fahrenheit")); m } } },
            ToolCall { function: ToolCallFunction { index: Some(1), name: "get_conditions".to_string(), arguments: { let mut m=HashMap::new(); m.insert("location".to_string(), json!("Tokyo")); m } } },
        ] },
        Case { name: "list multiple", inputs: vec!["[", "{", "\"name\": \"get_temperature\", ", "\"arguments\": {", "\"city\": \"London\"", "}", "},", "{", "\"name\": \"get_conditions\", ", "\"arguments\": {", "\"location\": \"Tokyo\"", "}", "}", "]"], tmpl: list_tmpl, content: "", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "get_temperature".to_string(), arguments: { let mut m=HashMap::new(); m.insert("city".to_string(), json!("London")); m } } },
            ToolCall { function: ToolCallFunction { index: Some(1), name: "get_conditions".to_string(), arguments: { let mut m=HashMap::new(); m.insert("location".to_string(), json!("Tokyo")); m } } },
        ] },
    ];

    for case in cases {
        let mut parser = Parser::new(case.tmpl, tools.clone());
        let mut calls = Vec::new();
        let mut content = String::new();
        for inp in case.inputs {
            let (c, cont) = parser.add(inp);
            calls.extend(c);
            content.push_str(&cont);
        }
        assert_eq!(content, case.content, "{}", case.name);
        assert_eq!(calls, case.calls, "{}", case.name);
    }
}

#[test]
fn parser_done_tests() {
    struct Case<'a> { name: &'a str, tag: &'a str, buffer: &'a str, want: bool }
    let cases = vec![
        Case { name: "empty", tag: "<tool_call>", buffer: "", want: false },
        Case { name: "json open", tag: "{", buffer: "{\"name\": \"get_weather\"", want: false },
        Case { name: "json closed", tag: "{", buffer: "{\"name\": \"get_weather\"}", want: true },
        Case { name: "json empty", tag: "{", buffer: "{}", want: true },
        Case { name: "list open", tag: "[", buffer: "[{\"name\": \"get_weather\"", want: false },
        Case { name: "list closed", tag: "[", buffer: "[{\"name\": \"get_weather\"}]", want: true },
    ];
    let tools = tools_list();
    for c in cases {
        let mut p = Parser::new_with_tag(tools.clone(), c.tag.to_string());
        p.set_buffer(c.buffer.as_bytes());
        assert_eq!(p.done(), c.want, "{}", c.name);
    }
}
