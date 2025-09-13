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
