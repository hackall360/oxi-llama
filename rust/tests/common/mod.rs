#![allow(dead_code)]
#![allow(unused_imports)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use api::client::Error as ApiError;
use api::{
    ChatRequest, ChatResponse, Client, EmbeddingRequest, EmbeddingResponse, GenerateRequest,
    GenerateResponse, ListResponse, Message, ProcessResponse, PullRequest, ShowRequest,
    ShowResponse,
};
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use once_cell::sync::Lazy;
use serde_json::json;
use tokio::time::timeout;
use url::Url;

static DEFAULT_MODEL: Lazy<String> = Lazy::new(|| {
    std::env::var("OLLAMA_TEST_SMOL_MODEL").unwrap_or_else(|_| "llama3.2:1b".to_string())
});

pub const BLUE_SKY_KEYWORDS: &[&str] = &[
    "rayleigh",
    "scatter",
    "scattering",
    "atmosphere",
    "nitrogen",
    "oxygen",
];

pub const DEFAULT_GENERATE_TIMEOUT: Duration = Duration::from_secs(120);

pub enum ModelStatus {
    Ready,
    Skipped,
}

pub struct GenerateOutcome {
    pub responses: Vec<GenerateResponse>,
    pub full_text: String,
    pub context: Vec<i32>,
}

impl GenerateOutcome {
    pub fn final_response(&self) -> Option<&GenerateResponse> {
        self.responses.last()
    }
}

pub struct ChatOutcome {
    pub responses: Vec<ChatResponse>,
    pub transcript: String,
}

impl ChatOutcome {
    pub fn final_message(&self) -> Option<&Message> {
        self.responses.last().map(|resp| &resp.message)
    }
}

#[derive(Clone)]
pub struct IntegrationTest {
    client: Client,
    base: Url,
    smol_model: String,
}

impl IntegrationTest {
    pub async fn new() -> Result<Option<Self>> {
        let client = match Client::from_env() {
            Ok(client) => client,
            Err(err) => return Err(to_anyhow(err).context("failed to create API client")),
        };
        let base = client.base().clone();
        match client.heartbeat().await {
            Ok(()) => Ok(Some(Self {
                client,
                base,
                smol_model: DEFAULT_MODEL.clone(),
            })),
            Err(ApiError::Http(err)) if err.is_connect() || err.is_timeout() => {
                println!(
                    "skipping integration test: unable to reach ollama server at {} ({err})",
                    base
                );
                Ok(None)
            }
            Err(err) => Err(to_anyhow(err).context("failed heartbeat")),
        }
    }

    pub fn client(&self) -> Client {
        self.client.clone()
    }

    pub fn smol_model(&self) -> &str {
        &self.smol_model
    }

    pub fn base_url(&self) -> &Url {
        &self.base
    }

    pub async fn ensure_model(&self, model: &str) -> Result<ModelStatus> {
        match self
            .client
            .show(&ShowRequest {
                model: model.to_string(),
                ..Default::default()
            })
            .await
        {
            Ok(_) => Ok(ModelStatus::Ready),
            Err(ApiError::Status(status)) if status.status_code == 404 => {
                if std::env::var("OLLAMA_SKIP_PULL").is_ok() {
                    println!(
                        "skipping test because model {model} is missing and OLLAMA_SKIP_PULL is set"
                    );
                    return Ok(ModelStatus::Skipped);
                }
                let pull_req = PullRequest {
                    model: model.to_string(),
                    name: model.to_string(),
                    stream: Some(true),
                    ..Default::default()
                };
                let last_update = Arc::new(Mutex::new(Instant::now()));
                let guard = last_update.clone();
                let pull_future = self.client.pull(&pull_req, move |_| {
                    if let Ok(mut ts) = guard.lock() {
                        *ts = Instant::now();
                    }
                    Ok(())
                });
                tokio::pin!(pull_future);
                let stall_limit = Duration::from_secs(60);
                loop {
                    let progress_deadline = {
                        let ts = last_update.lock().unwrap();
                        *ts + stall_limit
                    };
                    let wait_duration = progress_deadline.saturating_duration_since(Instant::now());
                    match timeout(wait_duration.max(Duration::from_secs(1)), &mut pull_future).await
                    {
                        Ok(result) => {
                            result.map_err(to_anyhow).context("pull failed")?;
                            break;
                        }
                        Err(_) => {
                            if Instant::now() >= progress_deadline {
                                return Err(anyhow!(
                                    "pull stalled for model {model}. set OLLAMA_SKIP_PULL=1 to skip"
                                ));
                            }
                        }
                    }
                }
                Ok(ModelStatus::Ready)
            }
            Err(err) => Err(to_anyhow(err).context(format!("failed to inspect model {model}"))),
        }
    }

    pub async fn generate(&self, mut req: GenerateRequest) -> Result<GenerateOutcome> {
        if req.model.is_empty() {
            req.model = self.smol_model.clone();
        }
        if req.stream.is_none() {
            req.stream = Some(true);
        }
        let mut responses = Vec::new();
        let mut aggregate = String::new();
        let mut context = Vec::new();
        let client = self.client.clone();
        timeout(
            DEFAULT_GENERATE_TIMEOUT,
            client.generate(&req, |resp| {
                context = resp.context.clone();
                aggregate.push_str(&resp.response);
                responses.push(resp);
                Ok(())
            }),
        )
        .await
        .context("generate timed out")?
        .map_err(to_anyhow)?;
        Ok(GenerateOutcome {
            responses,
            full_text: aggregate,
            context,
        })
    }

    pub async fn chat(&self, mut req: ChatRequest) -> Result<ChatOutcome> {
        if req.model.is_empty() {
            req.model = self.smol_model.clone();
        }
        if req.stream.is_none() {
            req.stream = Some(true);
        }
        let mut responses = Vec::new();
        let mut transcript = String::new();
        let client = self.client.clone();
        timeout(
            DEFAULT_GENERATE_TIMEOUT,
            client.chat(&req, |resp| {
                transcript.push_str(&resp.message.content);
                responses.push(resp);
                Ok(())
            }),
        )
        .await
        .context("chat timed out")?
        .map_err(to_anyhow)?;
        Ok(ChatOutcome {
            responses,
            transcript,
        })
    }

    pub async fn embeddings(&self, req: EmbeddingRequest) -> Result<EmbeddingResponse> {
        timeout(
            DEFAULT_GENERATE_TIMEOUT,
            self.client.clone().embeddings(&req),
        )
        .await
        .context("embeddings timed out")?
        .map_err(to_anyhow)
    }

    pub async fn show(&self, model: &str) -> Result<ShowResponse> {
        self.client
            .clone()
            .show(&ShowRequest {
                model: model.to_string(),
                ..Default::default()
            })
            .await
            .map_err(to_anyhow)
    }

    pub async fn list(&self) -> Result<ListResponse> {
        self.client.clone().list().await.map_err(to_anyhow)
    }

    pub async fn list_running(&self) -> Result<ProcessResponse> {
        self.client.clone().list_running().await.map_err(to_anyhow)
    }
}

pub fn build_generate_request(prompt: &str) -> GenerateRequest {
    let mut req = GenerateRequest::default();
    req.model = DEFAULT_MODEL.clone();
    req.prompt = prompt.to_string();
    req.stream = Some(true);
    req.keep_alive = Some(api::Duration(DEFAULT_GENERATE_TIMEOUT));
    req.options.insert("temperature".into(), json!(0));
    req.options.insert("seed".into(), json!(123));
    req
}

pub fn build_chat_request(prompt: &str) -> ChatRequest {
    let mut req = ChatRequest::default();
    req.model = DEFAULT_MODEL.clone();
    req.messages.push(Message {
        role: "user".into(),
        content: prompt.to_string(),
        ..Default::default()
    });
    req.stream = Some(true);
    req.keep_alive = Some(api::Duration(DEFAULT_GENERATE_TIMEOUT));
    req.options.insert("temperature".into(), json!(0));
    req.options.insert("seed".into(), json!(123));
    req
}

pub fn list_library_models() -> &'static [&'static str] {
    const MODELS: &[&str] = &[
        "gpt-oss:20b",
        "gemma3n:e2b",
        "mistral-small3.2:latest",
        "deepseek-r1:1.5b",
        "llama3.2-vision:latest",
        "qwen2.5-coder:latest",
        "qwen3:0.6b",
        "gemma3:1b",
    ];
    MODELS
}

pub fn list_embedding_models() -> &'static [&'static str] {
    const MODELS: &[&str] = &["all-minilm", "bge-large", "mxbai-embed-large"];
    MODELS
}

pub fn sample_image() -> Vec<u8> {
    // 1x1 transparent PNG
    const BASE64_IMAGE: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMBAAGoS0cAAAAASUVORK5CYII=";
    BASE64_STANDARD
        .decode(BASE64_IMAGE)
        .expect("invalid embedded image")
}

fn to_anyhow(err: ApiError) -> anyhow::Error {
    match err {
        ApiError::Http(err) => anyhow::Error::new(err),
        ApiError::Url(err) => anyhow::Error::new(err),
        ApiError::Json(err) => anyhow::Error::new(err),
        ApiError::Status(status) => anyhow!(
            "{} ({}): {}",
            status.status,
            status.status_code,
            status.error_message
        ),
    }
}
