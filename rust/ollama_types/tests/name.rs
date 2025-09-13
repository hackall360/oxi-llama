use std::path::PathBuf;
use ollama_types::model::{
    is_valid_namespace, parse_name, parse_name_bare, parse_name_from_filepath, Name,
};

fn join(parts: &[&str]) -> PathBuf {
    let mut pb = PathBuf::new();
    for p in parts {
        pb.push(p);
    }
    pb
}

#[test]
fn parse_name_parts() {
    struct Case<'a> {
        input: &'a str,
        want: Name,
        filepath: PathBuf,
    }
    let cases = vec![
        Case {
            input: "registry.ollama.ai/library/dolphin-mistral:7b-v2.6-dpo-laser-q6_K",
            want: Name {
                host: "registry.ollama.ai".into(),
                namespace: "library".into(),
                model: "dolphin-mistral".into(),
                tag: "7b-v2.6-dpo-laser-q6_K".into(),
            },
            filepath: join(&[
                "registry.ollama.ai",
                "library",
                "dolphin-mistral",
                "7b-v2.6-dpo-laser-q6_K",
            ]),
        },
        Case {
            input: "scheme://host:port/namespace/model:tag",
            want: Name {
                host: "host:port".into(),
                namespace: "namespace".into(),
                model: "model".into(),
                tag: "tag".into(),
            },
            filepath: join(&["host:port", "namespace", "model", "tag"]),
        },
        Case {
            input: "host/namespace/model:tag",
            want: Name {
                host: "host".into(),
                namespace: "namespace".into(),
                model: "model".into(),
                tag: "tag".into(),
            },
            filepath: join(&["host", "namespace", "model", "tag"]),
        },
        Case {
            input: "host/namespace/model",
            want: Name {
                host: "host".into(),
                namespace: "namespace".into(),
                model: "model".into(),
                tag: String::new(),
            },
            filepath: join(&["host", "namespace", "model", "latest"]),
        },
        Case {
            input: "namespace/model",
            want: Name {
                host: String::new(),
                namespace: "namespace".into(),
                model: "model".into(),
                tag: String::new(),
            },
            filepath: join(&["registry.ollama.ai", "namespace", "model", "latest"]),
        },
        Case {
            input: "model",
            want: Name {
                host: String::new(),
                namespace: String::new(),
                model: "model".into(),
                tag: String::new(),
            },
            filepath: join(&["registry.ollama.ai", "library", "model", "latest"]),
        },
    ];

    for case in cases {
        let got = parse_name_bare(case.input);
        assert_eq!(got, case.want, "parse_name_bare {}", case.input);
        let merged = parse_name(case.input);
        assert_eq!(merged.filepath(), case.filepath, "filepath for {}", case.input);
    }
}

#[test]
fn parse_name_from_filepath_test() {
    let cases: Vec<(String, Name)> = vec![
        (
            join(&["host", "namespace", "model", "tag"]).to_str().unwrap().to_string(),
            Name { host: "host".into(), namespace: "namespace".into(), model: "model".into(), tag: "tag".into() },
        ),
        (
            join(&["host:port", "namespace", "model", "tag"]).to_str().unwrap().to_string(),
            Name { host: "host:port".into(), namespace: "namespace".into(), model: "model".into(), tag: "tag".into() },
        ),
        (
            join(&["namespace", "model", "tag"]).to_str().unwrap().to_string(),
            Name::default(),
        ),
        (
            join(&["model", "tag"]).to_str().unwrap().to_string(),
            Name::default(),
        ),
    ];
    for (input, want) in cases {
        let got = parse_name_from_filepath(&input);
        assert_eq!(got, want, "{}", input);
    }
}

#[test]
fn display_shortest() {
    let cases = vec![
        ("registry.ollama.ai/library/model:latest", "model:latest"),
        ("registry.ollama.ai/library/model:tag", "model:tag"),
        ("registry.ollama.ai/namespace/model:tag", "namespace/model:tag"),
        ("host/namespace/model:tag", "host/namespace/model:tag"),
    ];
    for (input, want) in cases {
        let n = parse_name(input);
        assert_eq!(n.display_shortest(), want, "{}", input);
    }
}

#[test]
fn is_valid_namespace_cases() {
    let cases = vec![
        ("", false),
        ("a", true),
        ("a:b", false),
        ("a/b", false),
        ("himynameisjoe", true),
    ];
    for (input, expected) in cases {
        assert_eq!(is_valid_namespace(input), expected, "{}", input);
    }
}
