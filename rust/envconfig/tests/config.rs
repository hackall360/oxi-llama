use std::time::Duration;

use envconfig::{allowed_origins, host, keep_alive, load_timeout, var};

fn setenv(key: &str, val: &str) {
    unsafe { std::env::set_var(key, val) }
}
fn unset(key: &str) {
    unsafe { std::env::remove_var(key) }
}

#[test]
fn test_host() {
    let cases = [
        ("", "http://127.0.0.1:11434"),
        ("1.2.3.4", "http://1.2.3.4:11434"),
        (":1234", "http://:1234"),
        ("1.2.3.4:1234", "http://1.2.3.4:1234"),
        ("example.com", "http://example.com:11434"),
        ("example.com:1234", "http://example.com:1234"),
        (":0", "http://:0"),
        (":66000", "http://:11434"),
        (":-1", "http://:11434"),
        ("[::1]", "http://[::1]:11434"),
        ("[::]", "http://[::]:11434"),
        ("::1", "http://[::1]:11434"),
        ("[::1]:1337", "http://[::1]:1337"),
        (" 1.2.3.4 ", "http://1.2.3.4:11434"),
        ("\"1.2.3.4\"", "http://1.2.3.4:11434"),
        (" \" 1.2.3.4 \" ", "http://1.2.3.4:11434"),
        ("'1.2.3.4'", "http://1.2.3.4:11434"),
        ("http://1.2.3.4", "http://1.2.3.4:80"),
        ("http://1.2.3.4:4321", "http://1.2.3.4:4321"),
        ("https://1.2.3.4", "https://1.2.3.4:443"),
        ("https://1.2.3.4:4321", "https://1.2.3.4:4321"),
        (
            "https://example.com/ollama",
            "https://example.com:443/ollama",
        ),
    ];

    for (value, expect) in cases {
        setenv("OLLAMA_HOST", value);
        let actual = host();
        let actual = actual.trim_end_matches('/').to_string();
        assert_eq!(actual, expect, "case {value:?}");
    }
    unset("OLLAMA_HOST");
}

#[test]
fn test_origins() {
    let cases = [
        (
            "",
            vec![
                "http://localhost",
                "https://localhost",
                "http://localhost:*",
                "https://localhost:*",
                "http://127.0.0.1",
                "https://127.0.0.1",
                "http://127.0.0.1:*",
                "https://127.0.0.1:*",
                "http://0.0.0.0",
                "https://0.0.0.0",
                "http://0.0.0.0:*",
                "https://0.0.0.0:*",
                "app://*",
                "file://*",
                "tauri://*",
                "vscode-webview://*",
                "vscode-file://*",
            ],
        ),
        (
            "http://10.0.0.1",
            vec![
                "http://10.0.0.1",
                "http://localhost",
                "https://localhost",
                "http://localhost:*",
                "https://localhost:*",
                "http://127.0.0.1",
                "https://127.0.0.1",
                "http://127.0.0.1:*",
                "https://127.0.0.1:*",
                "http://0.0.0.0",
                "https://0.0.0.0",
                "http://0.0.0.0:*",
                "https://0.0.0.0:*",
                "app://*",
                "file://*",
                "tauri://*",
                "vscode-webview://*",
                "vscode-file://*",
            ],
        ),
        (
            "http://172.16.0.1,https://192.168.0.1",
            vec![
                "http://172.16.0.1",
                "https://192.168.0.1",
                "http://localhost",
                "https://localhost",
                "http://localhost:*",
                "https://localhost:*",
                "http://127.0.0.1",
                "https://127.0.0.1",
                "http://127.0.0.1:*",
                "https://127.0.0.1:*",
                "http://0.0.0.0",
                "https://0.0.0.0",
                "http://0.0.0.0:*",
                "https://0.0.0.0:*",
                "app://*",
                "file://*",
                "tauri://*",
                "vscode-webview://*",
                "vscode-file://*",
            ],
        ),
        (
            "http://totally.safe,http://definitely.legit",
            vec![
                "http://totally.safe",
                "http://definitely.legit",
                "http://localhost",
                "https://localhost",
                "http://localhost:*",
                "https://localhost:*",
                "http://127.0.0.1",
                "https://127.0.0.1",
                "http://127.0.0.1:*",
                "https://127.0.0.1:*",
                "http://0.0.0.0",
                "https://0.0.0.0",
                "http://0.0.0.0:*",
                "https://0.0.0.0:*",
                "app://*",
                "file://*",
                "tauri://*",
                "vscode-webview://*",
                "vscode-file://*",
            ],
        ),
    ];

    for (value, expect) in cases {
        setenv("OLLAMA_ORIGINS", value);
        assert_eq!(allowed_origins(), expect, "case {value}");
    }
    unset("OLLAMA_ORIGINS");
}

#[test]
fn test_keep_alive() {
    let cases = [
        ("", Duration::from_secs(5 * 60)),
        ("1s", Duration::from_secs(1)),
        ("1m", Duration::from_secs(60)),
        ("1h", Duration::from_secs(60 * 60)),
        ("5m0s", Duration::from_secs(5 * 60)),
        ("1h2m3s", Duration::from_secs(3723)),
        ("0", Duration::from_secs(0)),
        ("60", Duration::from_secs(60)),
        ("120", Duration::from_secs(120)),
        ("3600", Duration::from_secs(3600)),
        ("-0", Duration::from_secs(0)),
        ("-1", Duration::MAX),
        ("-1m", Duration::MAX),
        (" ", Duration::from_secs(5 * 60)),
        ("???", Duration::from_secs(5 * 60)),
        ("1d", Duration::from_secs(5 * 60)),
        ("1y", Duration::from_secs(5 * 60)),
        ("1w", Duration::from_secs(5 * 60)),
    ];
    for (value, expect) in cases {
        setenv("OLLAMA_KEEP_ALIVE", value);
        assert_eq!(keep_alive(), expect, "case {value}");
    }
    unset("OLLAMA_KEEP_ALIVE");
}

#[test]
fn test_load_timeout() {
    let default_timeout = Duration::from_secs(5 * 60);
    let cases = [
        ("", default_timeout),
        ("1s", Duration::from_secs(1)),
        ("1m", Duration::from_secs(60)),
        ("1h", Duration::from_secs(60 * 60)),
        ("5m0s", default_timeout),
        ("1h2m3s", Duration::from_secs(3723)),
        ("0", Duration::MAX),
        ("60", Duration::from_secs(60)),
        ("120", Duration::from_secs(120)),
        ("3600", Duration::from_secs(3600)),
        ("-0", Duration::MAX),
        ("-1", Duration::MAX),
        ("-1m", Duration::MAX),
        (" ", default_timeout),
        ("???", default_timeout),
        ("1d", default_timeout),
        ("1y", default_timeout),
        ("1w", default_timeout),
    ];
    for (value, expect) in cases {
        setenv("OLLAMA_LOAD_TIMEOUT", value);
        assert_eq!(load_timeout(), expect, "case {value}");
    }
    unset("OLLAMA_LOAD_TIMEOUT");
}

#[test]
fn test_var() {
    let cases = [
        ("value", "value"),
        (" value ", "value"),
        (" 'value' ", "value"),
        (" \"value\" ", "value"),
        (" ' value ' ", " value "),
        (" \" value \" ", " value "),
    ];
    for (input, expect) in cases {
        setenv("OLLAMA_VAR", input);
        assert_eq!(var("OLLAMA_VAR"), expect, "case {input}");
    }
    unset("OLLAMA_VAR");
}
