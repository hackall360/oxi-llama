use parser::thinking::Parser;

#[test]
fn extract_thinking() {
    let tests = vec![
        ("<think> internal </think> world", "internal ", "world"),
        ("<think>a</think><think>b</think>c", "a", "<think>b</think>c"),
        ("no think", "", "no think"),
    ];
    for (input, want_think, want_content) in tests {
        let mut p = Parser::default();
        p.opening_tag = "<think>".into();
        p.closing_tag = "</think>".into();
        let (t, c) = p.add_content(input);
        assert_eq!(t, want_think);
        assert_eq!(c, want_content);
    }
}

#[test]
fn thinking_streaming_basic() {
    let mut p = Parser::default();
    p.opening_tag = "<think>".into();
    p.closing_tag = "</think>".into();
    let (t, c) = p.add_content("<think>abc</think>\n\nhello");
    assert_eq!(t, "abc");
    assert_eq!(c, "hello");
}

#[test]
fn thinking_streaming_cases() {
    struct Step { input: &'static str, think: &'static str, content: &'static str }
    let cases: Vec<Vec<Step>> = vec![
        vec![
            Step { input: "  abc", think: "", content: "  abc" },
            Step { input: "def", think: "", content: "def" },
        ],
        vec![
            Step { input: "  <th", think: "", content: "" },
            Step { input: "in", think: "", content: "" },
            Step { input: "k>a", think: "a", content: "" },
        ],
        vec![
            Step { input: "<think>abc</th", think: "abc", content: "" },
            Step { input: "ink>def", think: "", content: "def" },
        ],
        vec![
            Step { input: "<think>abc</th", think: "abc", content: "" },
            Step { input: "ing>def", think: "</thing>def", content: "" },
            Step { input: "ghi</thi", think: "ghi", content: "" },
            Step { input: "nk>jkl", think: "", content: "jkl" },
        ],
        vec![
            Step { input: "  abc <think>def</think> ghi", think: "", content: "  abc <think>def</think> ghi" },
        ],
        vec![
            Step { input: "  <think>abc</think>", think: "abc", content: "" },
            Step { input: "\n\ndef", think: "", content: "def" },
        ],
    ];
    for case in cases {
        let mut p = Parser::default();
        p.opening_tag = "<think>".into();
        p.closing_tag = "</think>".into();
        for step in case {
            let (t, c) = p.add_content(step.input);
            assert_eq!(t, step.think);
            assert_eq!(c, step.content);
        }
    }
}
