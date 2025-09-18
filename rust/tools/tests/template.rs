use tools::parse_tag;

#[test]
fn parse_tag_tests() {
    struct Case<'a> {
        name: &'a str,
        template: &'a str,
        want: &'a str,
    }
    let cases = vec![
        Case {
            name: "empty",
            template: "",
            want: "{",
        },
        Case {
            name: "no tag",
            template: r#"{{if .ToolCalls}}{{end}}"#,
            want: "{",
        },
        Case {
            name: "no tag with range",
            template: r#"{{if .ToolCalls}}{{range .ToolCalls}}{{ . }}{{end}}{{end}}"#,
            want: "{",
        },
        Case {
            name: "tool call with json format",
            template: r#"{{if .ToolCalls}}```json
{{end}}"#,
            want: "```json",
        },
        Case {
            name: "square brackets",
            template: r#"{{if .ToolCalls}}[{{range .ToolCalls}}{{ . }}{{end}}]{{end}}"#,
            want: "[",
        },
        Case {
            name: "square brackets with whitespace",
            template: r#"{{if .ToolCalls}}
 [ {{range .ToolCalls}}{{ . }}{{end}}]{{end}}"#,
            want: "[",
        },
        Case {
            name: "tailing ]",
            template: r#"{{if .ToolCalls}}{{range .ToolCalls}}{{ . }}{{end}}]{{end}}"#,
            want: "{",
        },
        Case {
            name: "whitespace only",
            template: r#"{{if .ToolCalls}} {{range .ToolCalls}}{{ . }}{{end}}{{end}}"#,
            want: "{",
        },
        Case {
            name: "whitespace only in range",
            template: r#"{{if .ToolCalls}}{{range .ToolCalls}}
{{ . }}
{{end}}{{end}}"#,
            want: "{",
        },
        Case {
            name: "json objects",
            template: r#"{{if .ToolCalls}}{{range .ToolCalls}}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}{{end}}{{end}}"#,
            want: "{",
        },
        Case {
            name: "json objects with whitespace",
            template: r#"{{if .ToolCalls}}{{range .ToolCalls}}
{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}{{end}}{{end}}"#,
            want: "{",
        },
        Case {
            name: "json objects with CRLF",
            template: r#"{{if .ToolCalls}}{{range .ToolCalls}}

{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}{{end}}{{end}}"#,
            want: "{",
        },
        Case {
            name: "json objects with whitespace before and after range",
            template: r#"{{if .ToolCalls}}
{{range .ToolCalls}}
{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}

{{end}}

{{end}}"#,
            want: "{",
        },
        Case {
            name: "before and after range",
            template: r#"{{if .ToolCalls}}<|tool▁calls▁begin|>{{range .ToolCalls}}<|tool▁call▁begin|>functionget_current_weather
```json
{"location": "Tokyo"}
```<|tool▁call▁end|>
{{end}}<|tool▁calls▁end|>{{end}}"#,
            want: "<|tool▁calls▁begin|>",
        },
        Case {
            name: "after range",
            template: r#"{{if .ToolCalls}}{{range .ToolCalls}}<tool_call>{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}</tool_call>{{end}}{{end}}"#,
            want: "<tool_call>",
        },
        Case {
            name: "after range with leading whitespace before range",
            template: r#"{{if .ToolCalls}}
{{range .ToolCalls}}<tool_call>{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}</tool_call>{{end}}{{end}}"#,
            want: "<tool_call>",
        },
        Case {
            name: "tool call in range with {",
            template: r#"{{if .ToolCalls}}{{range .ToolCalls}}<tool_call>{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}<tool_call>{{end}}{{end}}"#,
            want: "<tool_call>",
        },
        Case {
            name: "tool call with multiple text nodes",
            template: r#"{{if .ToolCalls}}First text{{if .Something}}inner{{end}}Second text{{end}}"#,
            want: "First text",
        },
        Case {
            name: "action tag",
            template: r#"{{if .ToolCalls}}Action: ```json{{end}}"#,
            want: "Action: ```json",
        },
        Case {
            name: "incomplete functools bracket",
            template: r#"{{if .ToolCalls}}functools[{{end}}"#,
            want: "functools[",
        },
        Case {
            name: "uppercase tool call with incomplete bracket",
            template: r#"{{if .ToolCalls}}[TOOL_CALL] [{{end}}"#,
            want: "[TOOL_CALL] [",
        },
        Case {
            name: "uppercase tool call with adjacent bracket",
            template: r#"{{if .ToolCalls}}[TOOL_CALL][{{end}}"#,
            want: "[TOOL_CALL][",
        },
    ];
    for c in cases {
        let got = parse_tag(c.template);
        assert_eq!(got, c.want, "{}", c.name);
    }
}
