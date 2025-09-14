use api::client::Error;
use api::{ChatResponse, Client};
use reqwest::{Method, Url};
use serde::Deserialize;
use serde_json::json;
use std::thread;
use tiny_http::{Header, Response, Server};

#[tokio::test]
async fn client_from_environment() {
    struct Case {
        value: &'static str,
        expect: &'static str,
    }
    let cases = vec![
        Case {
            value: "",
            expect: "http://127.0.0.1:11434",
        },
        Case {
            value: "1.2.3.4",
            expect: "http://1.2.3.4:11434",
        },
        Case {
            value: ":1234",
            expect: "http://127.0.0.1:1234",
        },
        Case {
            value: "1.2.3.4:1234",
            expect: "http://1.2.3.4:1234",
        },
        Case {
            value: "http://1.2.3.4",
            expect: "http://1.2.3.4:80",
        },
        Case {
            value: "https://1.2.3.4",
            expect: "https://1.2.3.4:443",
        },
        Case {
            value: "https://1.2.3.4:1234",
            expect: "https://1.2.3.4:1234",
        },
        Case {
            value: "example.com",
            expect: "http://example.com:11434",
        },
        Case {
            value: "example.com:1234",
            expect: "http://example.com:1234",
        },
        Case {
            value: "http://example.com",
            expect: "http://example.com:80",
        },
        Case {
            value: "https://example.com",
            expect: "https://example.com:443",
        },
        Case {
            value: "https://example.com:1234",
            expect: "https://example.com:1234",
        },
        Case {
            value: "example.com/",
            expect: "http://example.com:11434",
        },
        Case {
            value: "example.com:1234/",
            expect: "http://example.com:1234",
        },
    ];
    for c in cases {
        std::env::set_var("OLLAMA_HOST", c.value);
        let client = Client::from_env().unwrap();
        let base = client.base();
        let expect = Url::parse(c.expect).unwrap();
        assert_eq!(base.scheme(), expect.scheme());
        assert_eq!(base.host_str(), expect.host_str());
        assert_eq!(base.port_or_known_default(), expect.port_or_known_default());
    }
}

fn start_server(body: String, status: u16, ct: &str) -> (Url, thread::JoinHandle<()>) {
    let server = Server::http("127.0.0.1:0").unwrap();
    let addr = format!("http://{}", server.server_addr());
    let ct_header = Header::from_bytes(&b"Content-Type"[..], ct.as_bytes()).unwrap();
    let handle = thread::spawn(move || {
        if let Ok(request) = server.recv() {
            let resp = Response::new(
                status.into(),
                vec![ct_header],
                body.as_bytes(),
                Some(body.len()),
                None,
            );
            let _ = request.respond(resp);
        }
    });
    (Url::parse(&addr).unwrap(), handle)
}

#[tokio::test]
async fn client_stream() {
    #[derive(Clone)]
    enum Resp {
        Chat(&'static str),
        Err(u16, &'static str),
    }
    let cases = vec![
        (
            vec![Resp::Err(400, "test error message")],
            Some("test error message"),
        ),
        (
            vec![Resp::Chat("{\"message\":{\"content\":\"partial response 1\"}}"),
                 Resp::Chat("{\"message\":{\"content\":\"partial response 2\"}}"),
                 Resp::Chat("{\"error\":\"mid-stream error\"}")],
            Some("mid-stream error"),
        ),
        (
            vec![Resp::Err(500, "custom error message")],
            Some("500"),
        ),
        (
            vec![Resp::Chat("{\"message\":{\"content\":\"chunk 1\"}}"),
                 Resp::Chat("{\"message\":{\"content\":\"chunk 2\"}}"),
                 Resp::Chat("{\"message\":{\"content\":\"final chunk\"},\"done\":true,\"done_reason\":\"stop\"}")],
            None,
        ),
    ];
    for (resps, want_err) in cases {
        let body = resps
            .iter()
            .map(|r| match r {
                Resp::Chat(s) => s.to_string(),
                Resp::Err(_, msg) => json!({"error": msg}).to_string(),
            })
            .collect::<Vec<String>>()
            .join("\n");
        let status = match resps.first().unwrap() {
            Resp::Err(code, _) => *code,
            _ => 200,
        };
        let (url, handle) = start_server(body.clone(), status, "application/x-ndjson");
        let client = Client::new(url, reqwest::Client::new());
        let mut chunks = Vec::new();
        let res = client
            .stream(Method::POST, "/v1/chat", None::<&()>, |b| {
                let resp: ChatResponse = serde_json::from_slice(b).unwrap();
                chunks.push(resp);
                Ok(())
            })
            .await;
        if let Some(_e) = want_err {
            assert!(res.is_err());
        } else {
            res.unwrap();
        }
        handle.join().unwrap();
    }
}

#[tokio::test]
async fn client_do() {
    #[derive(Clone)]
    enum Resp {
        Json(&'static str, u16),
    }
    let cases = vec![
        Resp::Json("{\"error\":\"test error message\"}", 400),
        Resp::Json("{\"error\":\"internal error\"}", 500),
        Resp::Json("{\"id\":\"msg_123\",\"success\":true}", 200),
    ];
    for resp in cases {
        let (body, status) = match resp {
            Resp::Json(b, s) => (b.to_string(), s),
        };
        let (url, handle) = start_server(body, status, "application/json");
        let client = Client::new(url, reqwest::Client::new());
        #[derive(Default, Deserialize)]
        struct R {
            id: Option<String>,
            success: Option<bool>,
        }
        let result: Result<R, Error> = client
            .do_request(Method::POST, "/v1/messages", None::<&()>)
            .await;
        if status != 200 {
            assert!(result.is_err());
        } else {
            let r = result.unwrap();
            assert_eq!(r.id.as_deref(), Some("msg_123"));
            assert_eq!(r.success, Some(true));
        }
        handle.join().unwrap();
    }
}
