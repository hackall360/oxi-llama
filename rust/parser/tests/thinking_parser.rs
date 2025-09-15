use parser::thinking::{Parser, State};

#[test]
fn extract_thinking() {
    let tests = vec![
        ("<think> internal </think> world", "internal ", "world"),
        (
            "<think>a</think><think>b</think>c",
            "a",
            "<think>b</think>c",
        ),
        ("no think", "", "no think"),
    ];

    for (input, want_think, want_content) in tests {
        let mut parser = Parser::default();
        parser.opening_tag = "<think>".into();
        parser.closing_tag = "</think>".into();
        let (thinking, content) = parser.add_content(input);
        assert_eq!(thinking, want_think);
        assert_eq!(content, want_content);
    }
}

#[test]
fn thinking_streaming() {
    struct Step {
        input: &'static str,
        want_thinking: &'static str,
        want_content: &'static str,
        want_state_after: State,
    }

    struct Case {
        desc: &'static str,
        steps: Vec<Step>,
    }

    let cases = vec![
        Case {
            desc: "content without a thinking tag",
            steps: vec![
                Step {
                    input: "  abc",
                    want_thinking: "",
                    want_content: "  abc",
                    want_state_after: State::ThinkingDone,
                },
                Step {
                    input: "def",
                    want_thinking: "",
                    want_content: "def",
                    want_state_after: State::ThinkingDone,
                },
            ],
        },
        Case {
            desc: "content before a thinking tag nerfs the thinking tag",
            steps: vec![Step {
                input: "  abc <think>def</think> ghi",
                want_thinking: "",
                want_content: "  abc <think>def</think> ghi",
                want_state_after: State::ThinkingDone,
            }],
        },
        Case {
            desc: "building up a thinking tag partially",
            steps: vec![
                Step {
                    input: "  <th",
                    want_thinking: "",
                    want_content: "",
                    want_state_after: State::LookingForOpening,
                },
                Step {
                    input: "in",
                    want_thinking: "",
                    want_content: "",
                    want_state_after: State::LookingForOpening,
                },
                Step {
                    input: "k>a",
                    want_thinking: "a",
                    want_content: "",
                    want_state_after: State::Thinking,
                },
            ],
        },
        Case {
            desc: "partial closing tag",
            steps: vec![
                Step {
                    input: "<think>abc</th",
                    want_thinking: "abc",
                    want_content: "",
                    want_state_after: State::Thinking,
                },
                Step {
                    input: "ink>def",
                    want_thinking: "",
                    want_content: "def",
                    want_state_after: State::ThinkingDone,
                },
            ],
        },
        Case {
            desc: "partial closing tag fakeout",
            steps: vec![
                Step {
                    input: "<think>abc</th",
                    want_thinking: "abc",
                    want_content: "",
                    want_state_after: State::Thinking,
                },
                Step {
                    input: "ing>def",
                    want_thinking: "</thing>def",
                    want_content: "",
                    want_state_after: State::Thinking,
                },
                Step {
                    input: "ghi</thi",
                    want_thinking: "ghi",
                    want_content: "",
                    want_state_after: State::Thinking,
                },
                Step {
                    input: "nk>jkl",
                    want_thinking: "",
                    want_content: "jkl",
                    want_state_after: State::ThinkingDone,
                },
            ],
        },
        Case {
            desc: "whitespace after thinking tag",
            steps: vec![Step {
                input: "  <think>abc</think>\n\ndef",
                want_thinking: "abc",
                want_content: "def",
                want_state_after: State::ThinkingDone,
            }],
        },
        Case {
            desc: "whitespace after thinking tag (incremental)",
            steps: vec![
                Step {
                    input: "  <think>abc</think>",
                    want_thinking: "abc",
                    want_content: "",
                    want_state_after: State::ThinkingDoneEatingWhitespace,
                },
                Step {
                    input: "\n\ndef",
                    want_thinking: "",
                    want_content: "def",
                    want_state_after: State::ThinkingDone,
                },
            ],
        },
        Case {
            desc: "whitespace after thinking tag with content and more whitespace",
            steps: vec![
                Step {
                    input: "  <think>abc</think>\n\ndef ",
                    want_thinking: "abc",
                    want_content: "def ",
                    want_state_after: State::ThinkingDone,
                },
                Step {
                    input: " ghi",
                    want_thinking: "",
                    want_content: " ghi",
                    want_state_after: State::ThinkingDone,
                },
            ],
        },
        Case {
            desc: "token by token",
            steps: vec![
                Step {
                    input: "<think>",
                    want_thinking: "",
                    want_content: "",
                    want_state_after: State::ThinkingStartedEatingWhitespace,
                },
                Step {
                    input: "\n",
                    want_thinking: "",
                    want_content: "",
                    want_state_after: State::ThinkingStartedEatingWhitespace,
                },
                Step {
                    input: "</think>",
                    want_thinking: "",
                    want_content: "",
                    want_state_after: State::ThinkingDoneEatingWhitespace,
                },
                Step {
                    input: "\n\n",
                    want_thinking: "",
                    want_content: "",
                    want_state_after: State::ThinkingDoneEatingWhitespace,
                },
                Step {
                    input: "Hi",
                    want_thinking: "",
                    want_content: "Hi",
                    want_state_after: State::ThinkingDone,
                },
                Step {
                    input: " there",
                    want_thinking: "",
                    want_content: " there",
                    want_state_after: State::ThinkingDone,
                },
            ],
        },
        Case {
            desc: "leading thinking whitespace",
            steps: vec![
                Step {
                    input: "  <think>   \t ",
                    want_thinking: "",
                    want_content: "",
                    want_state_after: State::ThinkingStartedEatingWhitespace,
                },
                Step {
                    input: "  these are some ",
                    want_thinking: "these are some ",
                    want_content: "",
                    want_state_after: State::Thinking,
                },
                Step {
                    input: "thoughts </think>  ",
                    want_thinking: "thoughts ",
                    want_content: "",
                    want_state_after: State::ThinkingDoneEatingWhitespace,
                },
                Step {
                    input: "  more content",
                    want_thinking: "",
                    want_content: "more content",
                    want_state_after: State::ThinkingDone,
                },
            ],
        },
    ];

    for case in cases {
        let mut parser = Parser::default();
        parser.opening_tag = "<think>".into();
        parser.closing_tag = "</think>".into();
        for (i, step) in case.steps.iter().enumerate() {
            let (thinking, content) = parser.add_content(step.input);
            assert_eq!(
                thinking, step.want_thinking,
                "case {} (step {}) thinking",
                case.desc, i
            );
            assert_eq!(
                content, step.want_content,
                "case {} (step {}) content",
                case.desc, i
            );
            assert_eq!(
                parser.state(),
                step.want_state_after,
                "case {} (step {}) state",
                case.desc,
                i
            );
        }
    }
}
