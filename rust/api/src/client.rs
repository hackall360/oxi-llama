use reqwest::{self, Method, Url};
use reqwest::header::USER_AGENT;
use url::ParseError;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::sync::OnceLock;

use version::{self, Version};

use crate::types::*;

#[derive(Clone)]
pub struct Client {
    base: Url,
    http: reqwest::Client,
}

#[derive(Debug)]
pub enum Error {
    Http(reqwest::Error),
    Url(ParseError),
    Json(serde_json::Error),
    Status(StatusError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Http(e) => write!(f, "{}", e),
            Error::Url(e) => write!(f, "{}", e),
            Error::Json(e) => write!(f, "{}", e),
            Error::Status(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for Error {}

impl From<reqwest::Error> for Error { fn from(e: reqwest::Error) -> Self { Error::Http(e) } }
impl From<ParseError> for Error { fn from(e: ParseError) -> Self { Error::Url(e) } }
impl From<serde_json::Error> for Error { fn from(e: serde_json::Error) -> Self { Error::Json(e) } }

impl Client {
    pub fn new(base: Url, http: reqwest::Client) -> Self { Self { base, http } }

    pub fn base(&self) -> &Url { &self.base }

    pub fn from_env() -> Result<Self, Error> {
        let host = std::env::var("OLLAMA_HOST").unwrap_or_default();
        let base = parse_host(&host).map_err(Error::Url)?;
        Ok(Client { base, http: reqwest::Client::new() })
    }

    pub async fn do_request<T, R>(&self, method: Method, path: &str, body: Option<&T>) -> Result<R, Error>
    where
        T: Serialize + ?Sized,
        R: DeserializeOwned + Default,
    {
        let url = self.base.join(path).map_err(Error::Url)?;
        let mut req = self.http.request(method, url);
        req = req.header(USER_AGENT, user_agent());
        if let Some(b) = body {
            req = req.json(b);
        }
        let resp = req.send().await.map_err(Error::Http)?;
        let status = resp.status();
        let bytes = resp.bytes().await.map_err(Error::Http)?;
        if status.is_client_error() || status.is_server_error() {
            let mut se: StatusError = serde_json::from_slice(&bytes).unwrap_or(StatusError {
                status_code: status.as_u16(),
                status: status.to_string(),
                error_message: String::from_utf8_lossy(&bytes).to_string(),
            });
            se.status_code = status.as_u16();
            return Err(Error::Status(se));
        }
        if bytes.is_empty() {
            return Ok(R::default());
        }
        let data: R = serde_json::from_slice(&bytes).map_err(Error::Json)?;
        Ok(data)
    }

    pub async fn stream<T, F>(&self, method: Method, path: &str, body: Option<&T>, mut f: F) -> Result<(), Error>
    where
        T: Serialize + ?Sized,
        F: FnMut(&[u8]) -> Result<(), Error>,
    {
        let url = self.base.join(path).map_err(Error::Url)?;
        let mut req = self.http.request(method, url);
        req = req.header(USER_AGENT, user_agent());
        if let Some(b) = body {
            req = req.json(b);
        }
        let resp = req.send().await.map_err(Error::Http)?;
        let status = resp.status();
        let bytes = resp.bytes().await.map_err(Error::Http)?;
        if status.is_client_error() || status.is_server_error() {
            let mut se: StatusError = serde_json::from_slice(&bytes).unwrap_or(StatusError {
                status_code: status.as_u16(),
                status: status.to_string(),
                error_message: String::from_utf8_lossy(&bytes).to_string(),
            });
            se.status_code = status.as_u16();
            return Err(Error::Status(se));
        }
        for line in bytes.split(|&b| b == b'\n') {
            if line.is_empty() { continue; }
            let v: Value = serde_json::from_slice(line).map_err(Error::Json)?;
            if let Some(e) = v.get("error").and_then(|e| e.as_str()) {
                if !e.is_empty() {
                    return Err(Error::Status(StatusError { status_code: status.as_u16(), status: status.to_string(), error_message: e.to_string() }));
                }
            }
            f(line)?;
        }
        Ok(())
    }

    pub async fn generate<F>(&self, req: &GenerateRequest, mut f: F) -> Result<(), Error>
    where
        F: FnMut(GenerateResponse) -> Result<(), Error>,
    {
        self.stream(Method::POST, "/api/generate", Some(req), |b| {
            let resp: GenerateResponse = serde_json::from_slice(b).map_err(Error::Json)?;
            f(resp)
        }).await
    }

    pub async fn chat<F>(&self, req: &ChatRequest, mut f: F) -> Result<(), Error>
    where
        F: FnMut(ChatResponse) -> Result<(), Error>,
    {
        self.stream(Method::POST, "/api/chat", Some(req), |b| {
            let resp: ChatResponse = serde_json::from_slice(b).map_err(Error::Json)?;
            f(resp)
        }).await
    }

    pub async fn pull<F>(&self, req: &PullRequest, mut f: F) -> Result<(), Error>
    where
        F: FnMut(ProgressResponse) -> Result<(), Error>,
    {
        self.stream(Method::POST, "/api/pull", Some(req), |b| {
            let resp: ProgressResponse = serde_json::from_slice(b).map_err(Error::Json)?;
            f(resp)
        }).await
    }

    pub async fn push<F>(&self, req: &PushRequest, mut f: F) -> Result<(), Error>
    where
        F: FnMut(ProgressResponse) -> Result<(), Error>,
    {
        self.stream(Method::POST, "/api/push", Some(req), |b| {
            let resp: ProgressResponse = serde_json::from_slice(b).map_err(Error::Json)?;
            f(resp)
        }).await
    }

    pub async fn create<F>(&self, req: &CreateRequest, mut f: F) -> Result<(), Error>
    where
        F: FnMut(ProgressResponse) -> Result<(), Error>,
    {
        self.stream(Method::POST, "/api/create", Some(req), |b| {
            let resp: ProgressResponse = serde_json::from_slice(b).map_err(Error::Json)?;
            f(resp)
        }).await
    }

    pub async fn list(&self) -> Result<ListResponse, Error> {
        self.do_request(Method::GET, "/api/tags", None::<&()>).await
    }

    pub async fn list_running(&self) -> Result<ProcessResponse, Error> {
        self.do_request(Method::GET, "/api/ps", None::<&()>).await
    }

    pub async fn copy(&self, req: &CopyRequest) -> Result<(), Error> {
        self.do_request::<_, Value>(Method::POST, "/api/copy", Some(req)).await.map(|_: Value| ())
    }

    pub async fn delete(&self, req: &DeleteRequest) -> Result<(), Error> {
        self.do_request::<_, Value>(Method::DELETE, "/api/delete", Some(req)).await.map(|_: Value| ())
    }

    pub async fn show(&self, req: &ShowRequest) -> Result<ShowResponse, Error> {
        self.do_request(Method::POST, "/api/show", Some(req)).await
    }

    pub async fn heartbeat(&self) -> Result<(), Error> {
        self.do_request::<_, Value>(Method::HEAD, "/", None::<&()>).await.map(|_: Value| ())
    }

    pub async fn embed(&self, req: &EmbedRequest) -> Result<EmbedResponse, Error> {
        self.do_request(Method::POST, "/api/embed", Some(req)).await
    }

    pub async fn embeddings(&self, req: &EmbeddingRequest) -> Result<EmbeddingResponse, Error> {
        self.do_request(Method::POST, "/api/embeddings", Some(req)).await
    }

    pub async fn create_blob<R: Into<reqwest::Body> + Send>(&self, digest: &str, body: R) -> Result<(), Error> {
        let url = self.base.join(&format!("/api/blobs/{}", digest)).map_err(Error::Url)?;
        let resp = self
            .http
            .post(url)
            .header(USER_AGENT, user_agent())
            .body(body)
            .send()
            .await
            .map_err(Error::Http)?;
        let status = resp.status();
        let bytes = resp.bytes().await.map_err(Error::Http)?;
        if status.is_client_error() || status.is_server_error() {
            let mut se: StatusError = serde_json::from_slice(&bytes).unwrap_or(StatusError {
                status_code: status.as_u16(),
                status: status.to_string(),
                error_message: String::from_utf8_lossy(&bytes).to_string(),
            });
            se.status_code = status.as_u16();
            return Err(Error::Status(se));
        }
        Ok(())
    }

    pub async fn version(&self) -> Result<String, Error> {
        #[derive(Deserialize, Default)]
        struct V { version: String }
        let v: V = self.do_request(Method::GET, "/api/version", None::<&()>).await?;
        Ok(v.version)
    }

    // Sync wrappers
    pub fn generate_blocking<F>(&self, req: &GenerateRequest, f: F) -> Result<(), Error>
    where
        F: FnMut(GenerateResponse) -> Result<(), Error> + Send + 'static,
    {
        tokio::runtime::Runtime::new().unwrap().block_on(self.generate(req, f))
    }

    pub fn chat_blocking<F>(&self, req: &ChatRequest, f: F) -> Result<(), Error>
    where
        F: FnMut(ChatResponse) -> Result<(), Error> + Send + 'static,
    {
        tokio::runtime::Runtime::new().unwrap().block_on(self.chat(req, f))
    }
}

fn user_agent() -> &'static str {
    static USER_AGENT_VALUE: OnceLock<String> = OnceLock::new();
    USER_AGENT_VALUE
        .get_or_init(|| {
            let meta = version::metadata();
            let mut version = Version.to_string();
            if meta.git_dirty {
                version.push_str("-dirty");
            }
            let commit = meta
                .git_commit
                .get(..7)
                .filter(|s| !s.is_empty())
                .unwrap_or(meta.git_commit)
                .to_string();
            format!(
                "ollama/{version} ({arch} {os}; commit {commit}) Rust/{rustc}",
                arch = std::env::consts::ARCH,
                os = std::env::consts::OS,
                commit = commit,
                rustc = meta.rustc_version,
            )
        })
        .as_str()
}

fn parse_host(input: &str) -> Result<Url, ParseError> {
    let mut default_port = "11434".to_string();
    let s = input.trim();
    let (scheme, rest) = if let Some((sch, hp)) = s.split_once("://") {
        if sch == "http" { default_port = "80".into(); }
        else if sch == "https" { default_port = "443".into(); }
        (sch.to_string(), hp.to_string())
    } else {
        ("http".into(), s.to_string())
    };
    let (hostport, _path) = if let Some((hp, p)) = rest.split_once('/') {
        (hp.to_string(), p)
    } else {
        (rest.to_string(), "")
    };
    let mut host = "127.0.0.1".to_string();
    let mut port = default_port.clone();
    if let Some(idx) = hostport.rfind(':') {
        host = hostport[..idx].to_string();
        port = hostport[idx+1..].to_string();
    } else if !hostport.is_empty() {
        host = hostport;
    }
    if host.is_empty() { host = "127.0.0.1".to_string(); }
    if port.parse::<i64>().is_err() {
        port = default_port;
    }
    Url::parse(&format!("{}://{}:{}", scheme, host, port))
}
