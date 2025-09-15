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
fn parser_cases() {
    let tools = tools_list();
    let qwen = "{{if .ToolCalls}}<tool_call>{{range .ToolCalls}}{\"name\": \"{{.Function.Name}}\",\"arguments\": {{.Function.Arguments}}}{{end}}</tool_call>{{end}}";
    let deepseek = "{{if .ToolCalls}}<|tool▁calls▁begin|>{{range .ToolCalls}}<|tool▁call▁begin|>function<|tool▁sep|>get_current_weather\n```json\n{\"location\": \"Tokyo\"}\n```<|tool▁call▁end|>{{end}}<|tool▁calls▁end|><|end▁of▁sentence|>{{end}}";
    let json_tmpl = "{{if .ToolCalls}}{{range .ToolCalls}}{\"name\": \"{{.Function.Name}}\", \"arguments\": {{.Function.Arguments}}}{{end}}{{end}}";
    let list_tmpl = "{{if .ToolCalls}}[{{range .ToolCalls}}{\"name\": \"{{.Function.Name}}\", \"arguments\": {{.Function.Arguments}}}{{end}}]{{end}}";
    let mistral = "{{if .ToolCalls}}[TOOL_CALLS] [{{range .ToolCalls}}{\"name\": \"{{.Function.Name}}\", \"arguments\": {{.Function.Arguments}}}{{end}}][/TOOL_CALLS]{{end}}";

    struct Case<'a> {
        name: &'a str,
        inputs: Vec<&'a str>,
        tmpl: &'a str,
        content: &'a str,
        calls: Vec<ToolCall>,
    }

    let cases: Vec<Case> = vec![
        Case { name: "no tool calls - just text", inputs: vec!["Hello, how can I help you today?"], tmpl: qwen, content: "Hello, how can I help you today?", calls: vec![] },
        Case { name: "empty input", inputs: vec![""], tmpl: qwen, content: "", calls: vec![] },
        Case { name: "tool call", inputs: vec!["<tool_call>{\"name\": \"get_conditions\", \"arguments\": {\"location\": \"San Francisco\"}}</tool_call>"], tmpl: qwen, content: "", calls: vec![ToolCall { function: ToolCallFunction { index: Some(0), name: "get_conditions".to_string(), arguments: { let mut m=HashMap::new(); m.insert("location".to_string(), json!("San Francisco")); m } } }] },
        Case { name: "empty args", inputs: vec!["<tool_call>{\"name\": \"get_conditions\", \"arguments\": {}}</tool_call>"], tmpl: qwen, content: "", calls: vec![ToolCall { function: ToolCallFunction { index: Some(0), name: "get_conditions".to_string(), arguments: HashMap::new() } }] },
        Case { name: "text before tool call", inputs: vec!["Let me check the weather. <tool_call>{\"name\": \"get_temperature\", \"arguments\": {\"city\": \"New York\"}}</tool_call>"], tmpl: qwen, content: "Let me check the weather. ", calls: vec![ToolCall { function: ToolCallFunction { index: Some(0), name: "get_temperature".to_string(), arguments: { let mut m=HashMap::new(); m.insert("city".to_string(), json!("New York")); m } } }] },
        Case { name: "qwen no args with text", inputs: vec!["Let me say hello to the user. I'll use the say_hello tool. "], tmpl: qwen, content: "Let me say hello to the user. I'll use the say_hello tool. ", calls: vec![] },
        Case { name: "two tool calls in a list", inputs: vec!["[TOOL_CALLS] [{\"name\": \"get_temperature\", \"arguments\": {\"city\": \"London\", \"format\": \"fahrenheit\"}}, {\"name\": \"get_conditions\", \"arguments\": {\"location\": \"Tokyo\"}}][/TOOL_CALLS]"], tmpl: mistral, content: "", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "get_temperature".to_string(), arguments: { let mut m=HashMap::new(); m.insert("city".to_string(), json!("London")); m.insert("format".to_string(), json!("fahrenheit")); m } } },
            ToolCall { function: ToolCallFunction { index: Some(1), name: "get_conditions".to_string(), arguments: { let mut m=HashMap::new(); m.insert("location".to_string(), json!("Tokyo")); m } } },
        ] },
        Case { name: "qwen two tool calls", inputs: vec!["Okay, let's call both tools! <tool_call>{\"name\": \"get_temperature\", \"arguments\": {\"city\": \"London\", \"format\": \"fahrenheit\"}}</tool_call><tool_call>{\"name\": \"get_conditions\", \"arguments\": {\"location\": \"Tokyo\"}}</tool_call>"], tmpl: qwen, content: "Okay, let's call both tools! ", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "get_temperature".to_string(), arguments: { let mut m=HashMap::new(); m.insert("city".to_string(), json!("London")); m.insert("format".to_string(), json!("fahrenheit")); m } } },
            ToolCall { function: ToolCallFunction { index: Some(1), name: "get_conditions".to_string(), arguments: { let mut m=HashMap::new(); m.insert("location".to_string(), json!("Tokyo")); m } } },
        ] },
        Case { name: "empty args followed by args", inputs: vec!["Let me say hello and check the weather. <tool_call>{\"name\": \"say_hello\", \"arguments\": {}}</tool_call><tool_call>{\"name\": \"get_temperature\", \"arguments\": {\"city\": \"London\", \"format\": \"fahrenheit\"}}</tool_call>"], tmpl: qwen, content: "Let me say hello and check the weather. ", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "say_hello".to_string(), arguments: HashMap::new() } },
            ToolCall { function: ToolCallFunction { index: Some(1), name: "get_temperature".to_string(), arguments: { let mut m=HashMap::new(); m.insert("city".to_string(), json!("London")); m.insert("format".to_string(), json!("fahrenheit")); m } } },
        ] },
        Case { name: "qwen empty followed by args", inputs: vec!["Let me check the weather. <tool_call>{\"name\": \"get_conditions\", \"arguments\": {}}</tool_call><tool_call>{\"name\": \"get_conditions\", \"arguments\": {\"location\": \"Tokyo\"}}"], tmpl: qwen, content: "Let me check the weather. ", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "get_conditions".to_string(), arguments: HashMap::new() } },
            ToolCall { function: ToolCallFunction { index: Some(1), name: "get_conditions".to_string(), arguments: { let mut m=HashMap::new(); m.insert("location".to_string(), json!("Tokyo")); m } } },
        ] },
        Case { name: "deepseek", inputs: vec!["<think>Wait, I need to call a tool</think><|tool▁calls▁begin|><|tool▁call▁begin|>function<|tool▁sep|>get_temperature\n```json\n{\"city\": \"Tokyo\"}\n```<|tool▁call▁end|><|tool▁calls▁end|><|end▁of▁sentence|>"], tmpl: deepseek, content: "<think>Wait, I need to call a tool</think>", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "get_temperature".to_string(), arguments: { let mut m=HashMap::new(); m.insert("city".to_string(), json!("Tokyo")); m } } },
        ] },
        Case { name: "deepseek incremental", inputs: vec!["<think>Wait", ", I need", " to call", " a tool</think><|too", "l▁calls▁begin|>", "<|tool▁call▁begin|>function<|tool▁sep|>get_temperature\n", "```json\n", "{\"city\": \"Tokyo\"}\n", "```", "<|tool▁call▁end|>", "<|tool▁calls▁end|>", "<|end▁of▁sentence|>"], tmpl: deepseek, content: "<think>Wait, I need to call a tool</think>", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "get_temperature".to_string(), arguments: { let mut m=HashMap::new(); m.insert("city".to_string(), json!("Tokyo")); m } } },
        ] },
        Case { name: "json", inputs: vec!["{", "\"name\": \"get_temperature\",", "\"arguments\": {", "\"city\": \"Tokyo\"", "}", "}"], tmpl: json_tmpl, content: "", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "get_temperature".to_string(), arguments: { let mut m=HashMap::new(); m.insert("city".to_string(), json!("Tokyo")); m } } },
        ] },
        Case { name: "json maybe a tool call", inputs: vec!["{", "\"name\": \"get_temperature\",", "\"arguments\": {"], tmpl: json_tmpl, content: "", calls: vec![] },
        Case { name: "json not a tool call", inputs: vec!["{", "\"name\": \"search\", ", "\"arguments\": {", "\"query\": \"What is the capital of Canada?\"", "}", "}"], tmpl: json_tmpl, content: "{\"name\": \"search\", \"arguments\": {\"query\": \"What is the capital of Canada?\"}}", calls: vec![] },
        Case { name: "json object followed by tool call", inputs: vec!["{\"name\": \"jeff\"}", "{\"name\": \"get_conditions\", \"arguments\": {\"location\": \"San Francisco\"}}"], tmpl: json_tmpl, content: "{\"name\": \"jeff\"}{\"name\": \"get_conditions\", \"arguments\": {\"location\": \"San Francisco\"}}", calls: vec![] },
        Case { name: "json object followed by tool call split", inputs: vec!["{\"name\": \"jeff\"} {", "\"name\": \"get_conditions\", \"arguments\": {\"location\": \"San Francisco\"}}"], tmpl: json_tmpl, content: "{\"name\": \"jeff\"} {\"name\": \"get_conditions\", \"arguments\": {\"location\": \"San Francisco\"}}", calls: vec![] },
        Case { name: "json code", inputs: vec!["for { fmt.Println(\"hello\") }"], tmpl: json_tmpl, content: "for { fmt.Println(\"hello\") }", calls: vec![] },
        Case { name: "list multiple", inputs: vec!["[", "{", "\"name\": \"get_temperature\", ", "\"arguments\": {", "\"city\": \"London\"", "}", "},", "{", "\"name\": \"get_conditions\", ", "\"arguments\": {", "\"location\": \"Tokyo\"", "}", "}]"], tmpl: list_tmpl, content: "", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "get_temperature".to_string(), arguments: { let mut m=HashMap::new(); m.insert("city".to_string(), json!("London")); m } } },
            ToolCall { function: ToolCallFunction { index: Some(1), name: "get_conditions".to_string(), arguments: { let mut m=HashMap::new(); m.insert("location".to_string(), json!("Tokyo")); m } } },
        ] },
        Case { name: "list partial", inputs: vec!["[{", "\"name\": \"get_conditions\", ", "\"arguments\": {", "\"location\": \"Tokyo\"", "}", "}"], tmpl: list_tmpl, content: "", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "get_conditions".to_string(), arguments: { let mut m=HashMap::new(); m.insert("location".to_string(), json!("Tokyo")); m } } },
        ] },
        Case { name: "list invalid", inputs: vec!["[", "{", "\"name\": \"search\", ", "\"arguments\": {", "\"query\": \"What is the capital of Canada?\"", "}", "}"], tmpl: list_tmpl, content: "", calls: vec![] },
        Case { name: "list trailing ]", inputs: vec!["[", "{", "\"name\": \"get_conditions\", ", "\"arguments\": {", "\"location\": \"Tokyo\"", "}", "}", "]", "]"], tmpl: list_tmpl, content: "", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "get_conditions".to_string(), arguments: { let mut m=HashMap::new(); m.insert("location".to_string(), json!("Tokyo")); m } } },
        ] },
        Case { name: "list not a tool call", inputs: vec!["[special", " del", "ivery]"], tmpl: list_tmpl, content: "[special delivery]", calls: vec![] },
        Case { name: "tool name with collision", inputs: vec!["<tool_call>", "{", "\"name\": \"say_hello", "_world\",", "\"arguments\": {}}", "}"], tmpl: qwen, content: "", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "say_hello_world".to_string(), arguments: HashMap::new() } },
        ] },
        Case { name: "tool name with collision multiple", inputs: vec!["<tool_call>", "{", "\"name\": \"say_hello", "_world\",", "\"arguments\": {}}", "</tool_call>", "<tool_call>", "{", "\"name\": \"say_hello\",", "\"arguments\": {}}", "</tool_call>"], tmpl: qwen, content: "", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "say_hello_world".to_string(), arguments: HashMap::new() } },
            ToolCall { function: ToolCallFunction { index: Some(1), name: "say_hello".to_string(), arguments: HashMap::new() } },
        ] },
        Case { name: "tool name with collision non streaming", inputs: vec!["<tool_call>{\"name\": \"say_hello"], tmpl: qwen, content: "", calls: vec![] },
        Case { name: "tool name with collision non streaming multiple", inputs: vec!["<tool_call>{\"name\": \"say_hello\", \"arguments\": {}}</tool_call><tool_call>{\"name\": \"say_hello_world\", \"arguments\": {}}"], tmpl: qwen, content: "", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "say_hello".to_string(), arguments: HashMap::new() } },
            ToolCall { function: ToolCallFunction { index: Some(1), name: "say_hello_world".to_string(), arguments: HashMap::new() } },
        ] },
        Case { name: "tool name with collision non streaming shorter", inputs: vec!["<tool_call>{\"name\": \"say_hello\", \"arguments\": {}}</tool_call>"], tmpl: qwen, content: "", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "say_hello".to_string(), arguments: HashMap::new() } },
        ] },
        Case { name: "tool name with collision non streaming longer", inputs: vec!["<tool_call>{\"name\": \"say_hello_world\", \"arguments\": {}}</tool_call>"], tmpl: qwen, content: "", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "say_hello_world".to_string(), arguments: HashMap::new() } },
        ] },
        Case { name: "tool name with substring of another json", inputs: vec!["{", "\"name\": \"get_address\",", "\"arguments\": {", "\"location\": \"London\"", "}", "}"], tmpl: json_tmpl, content: "", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "get_address".to_string(), arguments: { let mut m=HashMap::new(); m.insert("location".to_string(), json!("London")); m } } },
        ] },
        Case { name: "tool name with substring of another", inputs: vec!["<tool_call>{\"name\": \"get_address\", \"arguments\": {\"location\": \"London\"}}</tool_call>"], tmpl: qwen, content: "", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "get_address".to_string(), arguments: { let mut m=HashMap::new(); m.insert("location".to_string(), json!("London")); m } } },
        ] },
        Case { name: "args before name", inputs: vec!["<tool_call>{\"arguments\": {\"a\": \"5\", \"b\": \"10\"}, \"name\": \"add\"}</tool_call>"], tmpl: qwen, content: "", calls: vec![
            ToolCall { function: ToolCallFunction { index: Some(0), name: "add".to_string(), arguments: { let mut m=HashMap::new(); m.insert("a".to_string(), json!("5")); m.insert("b".to_string(), json!("10")); m } } },
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
        Case { name: "list empty", tag: "[", buffer: "[]", want: true },
    ];
    let tools = tools_list();
    for c in cases {
        let mut p = Parser::new_with_tag(tools.clone(), c.tag.to_string());
        p.set_buffer(c.buffer.as_bytes());
        assert_eq!(p.done(), c.want, "{}", c.name);
    }
}

