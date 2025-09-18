use ollama_types::model::{
    is_valid_namespace, parse_name, parse_name_bare, parse_name_from_filepath, Name,
};
use std::path::PathBuf;

const PART80: &str =
    "88888888888888888888888888888888888888888888888888888888888888888888888888888888";

fn join(parts: &[&str]) -> PathBuf {
    let mut pb = PathBuf::new();
    for p in parts {
        pb.push(p);
    }
    pb
}

#[test]
fn parse_name_parts() {
    let part350 = "3".repeat(350);
    struct Case {
        input: String,
        want: Name,
        filepath: PathBuf,
    }
    let cases = vec![
        Case {
            input: "registry.ollama.ai/library/dolphin-mistral:7b-v2.6-dpo-laser-q6_K".into(),
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
            input: "scheme://host:port/namespace/model:tag".into(),
            want: Name {
                host: "host:port".into(),
                namespace: "namespace".into(),
                model: "model".into(),
                tag: "tag".into(),
            },
            filepath: join(&["host:port", "namespace", "model", "tag"]),
        },
        Case {
            input: "host/namespace/model:tag".into(),
            want: Name {
                host: "host".into(),
                namespace: "namespace".into(),
                model: "model".into(),
                tag: "tag".into(),
            },
            filepath: join(&["host", "namespace", "model", "tag"]),
        },
        Case {
            input: "host/namespace/model".into(),
            want: Name {
                host: "host".into(),
                namespace: "namespace".into(),
                model: "model".into(),
                tag: String::new(),
            },
            filepath: join(&["host", "namespace", "model", "latest"]),
        },
        Case {
            input: "namespace/model".into(),
            want: Name {
                host: String::new(),
                namespace: "namespace".into(),
                model: "model".into(),
                tag: String::new(),
            },
            filepath: join(&["registry.ollama.ai", "namespace", "model", "latest"]),
        },
        Case {
            input: "model".into(),
            want: Name {
                host: String::new(),
                namespace: String::new(),
                model: "model".into(),
                tag: String::new(),
            },
            filepath: join(&["registry.ollama.ai", "library", "model", "latest"]),
        },
        Case {
            input: format!("{PART80}/{PART80}/{PART80}:{PART80}"),
            want: Name {
                host: PART80.into(),
                namespace: PART80.into(),
                model: PART80.into(),
                tag: PART80.into(),
            },
            filepath: join(&[PART80, PART80, PART80, PART80]),
        },
        Case {
            input: format!("{part350}/{PART80}/{PART80}:{PART80}"),
            want: Name {
                host: part350.clone(),
                namespace: PART80.into(),
                model: PART80.into(),
                tag: PART80.into(),
            },
            filepath: join(&[part350.as_str(), PART80, PART80, PART80]),
        },
    ];

    for case in cases {
        let got = parse_name_bare(&case.input);
        assert_eq!(got, case.want, "parse_name_bare {}", case.input);
        let merged = parse_name(&case.input);
        assert_eq!(
            merged.filepath(),
            case.filepath,
            "filepath for {}",
            case.input
        );
    }
}

#[test]
fn parse_name_from_filepath_test() {
    let cases: Vec<(String, Name)> = vec![
        (
            join(&["host", "namespace", "model", "tag"])
                .to_str()
                .unwrap()
                .to_string(),
            Name {
                host: "host".into(),
                namespace: "namespace".into(),
                model: "model".into(),
                tag: "tag".into(),
            },
        ),
        (
            join(&["host:port", "namespace", "model", "tag"])
                .to_str()
                .unwrap()
                .to_string(),
            Name {
                host: "host:port".into(),
                namespace: "namespace".into(),
                model: "model".into(),
                tag: "tag".into(),
            },
        ),
        (
            join(&["namespace", "model", "tag"])
                .to_str()
                .unwrap()
                .to_string(),
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
        (
            "registry.ollama.ai/namespace/model:tag",
            "namespace/model:tag",
        ),
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

#[test]
fn parse_name_default() {
    let n = parse_name("xx");
    assert_eq!(n.to_string(), "registry.ollama.ai/library/xx:latest");
}

#[test]
fn name_is_valid() {
    let part350 = "3".repeat(350);
    let test_cases: Vec<(String, bool)> = vec![
        ("".into(), false),
        ("_why/_the/_lucky:_stiff".into(), true),
        ("h/n/m:t".into(), true),
        ("host/namespace/model:tag".into(), true),
        ("host/namespace/model".into(), false),
        ("namespace/model".into(), false),
        ("model".into(), false),
        (format!("{PART80}/{PART80}/{PART80}:{PART80}"), true),
        (format!("{part350}/{PART80}/{PART80}:{PART80}"), true),
        ("h/nn/mm:t".into(), true),
        ("m".into(), false),
        ("n/m:".into(), false),
        ("h/n/m".into(), false),
        ("@t".into(), false),
        ("m@d".into(), false),
        ("^".into(), false),
        ("mm:".into(), false),
        ("/nn/mm".into(), false),
        ("//".into(), false),
        ("//mm".into(), false),
        ("hh//".into(), false),
        ("//mm:@".into(), false),
        ("00@".into(), false),
        ("@".into(), false),
        ("-hh/nn/mm:tt".into(), false),
        ("hh/-nn/mm:tt".into(), false),
        ("hh/nn/-mm:tt".into(), false),
        ("hh/nn/mm:-tt".into(), false),
        ("host:https/namespace/model:tag".into(), true),
        ("host/name:space/model:tag".into(), false),
    ];

    let mut tested_string = false;
    for (s, want) in test_cases {
        let n = parse_name_bare(&s);
        let got = n.is_valid();
        assert_eq!(got, want, "is_valid {}", s);
        if got {
            assert_eq!(parse_name_bare(&s).to_string(), s);
            tested_string = true;
        }
    }
    assert!(tested_string, "no tests for Name::to_string");
}
