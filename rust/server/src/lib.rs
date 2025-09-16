use axum::{routing::{post, delete, get}, Router, Json, extract::{State, Path, Request as AxumRequest}, middleware::{from_fn, Next}, http::StatusCode, response::Response};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use serde_json::{json, Value};
use version::{Version, BUILD_METADATA};
use api::{CreateRequest, GenerateRequest, DeleteRequest, GenerateResponse, ChatResponse, EmbedResponse, ListResponse, ShowResponse, Message as ApiMessage, ListModelResponse};
use api::openai::{self, ChatCompletionRequest, CompletionRequest, EmbedRequest};
use time::OffsetDateTime;
use convert::{convert_model, ModelFormat};
use std::path::Path as StdPath;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use ed25519_dalek::{VerifyingKey, Signature, Verifier};
#[derive(Default)]
pub struct AppState {
    models: Mutex<HashMap<String, ListModelResponse>>,
}

pub fn app() -> Router<Arc<AppState>> {
    let state = Arc::new(AppState::default());
    Router::new()
        .route("/api/create", post(create_handler))
        .route("/api/generate", post(generate_handler))
        .route("/api/delete", delete(delete_handler))
        .route("/api/version", get(version_handler).head(version_handler))
        .route("/api/tags", get(list_handler))
        .route("/v1/chat/completions", post(openai_chat_handler))
        .route("/v1/completions", post(openai_completion_handler))
        .route("/v1/embeddings", post(openai_embeddings_handler))
        .route("/v1/models", get(openai_models_handler))
        .route("/v1/models/:model", get(openai_model_handler))
        .with_state(state)
        .layer(from_fn(auth_middleware))
}

pub async fn create_handler(State(state): State<Arc<AppState>>, Json(req): Json<CreateRequest>) -> Json<Value> {
    let name = if !req.model.is_empty() { req.model.clone() } else { req.name.clone() };
    let _ = convert_model(StdPath::new(&name), StdPath::new("model.bin"), ModelFormat::LLaMA);
    let now = OffsetDateTime::now_utc().format(&time::format_description::well_known::Rfc3339).unwrap_or_default();
    let mut models = state.models.lock().unwrap();
    models.insert(name.clone(), ListModelResponse { name: name.clone(), model: name, modified_at: now, size: 0, digest: String::new(), ..Default::default() });
    Json(json!({"status": "ok"}))
}

pub async fn generate_handler(State(_state): State<Arc<AppState>>, Json(_req): Json<GenerateRequest>) -> Json<GenerateResponse> {
    // Return a minimal response compatible with API types.
    Json(GenerateResponse{
        response: "not implemented".into(),
        done: true,
        ..Default::default()
    })
}

pub async fn delete_handler(State(state): State<Arc<AppState>>, Json(req): Json<DeleteRequest>) -> Json<Value> {
    let name = if !req.model.is_empty() { req.model } else { req.name };
    let mut models = state.models.lock().unwrap();
    models.remove(&name);
    Json(json!({"status": "ok"}))
}

pub async fn list_handler(State(state): State<Arc<AppState>>) -> Json<ListResponse> {
    let models = state.models.lock().unwrap();
    Json(ListResponse { models: models.values().cloned().collect() })
}

pub async fn version_handler() -> Json<Value> {
    let meta = BUILD_METADATA;
    Json(json!({
        "version": Version,
        "git_commit": meta.git_commit,
        "git_dirty": meta.git_dirty,
        "build_timestamp": meta.build_timestamp,
        "build_target": meta.build_target,
        "build_profile": meta.build_profile,
        "rustc_version": meta.rustc_version,
    }))
}

pub async fn openai_chat_handler(Json(req): Json<ChatCompletionRequest>) -> Json<openai::ChatCompletion> {
    let model = req.model.clone();
    let now = OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_default();
    let resp = ChatResponse {
        model,
        created_at: now,
        message: ApiMessage { role: "assistant".into(), content: "not implemented".into(), ..Default::default() },
        done_reason: String::new(),
        done: true,
        metrics: Default::default(),
    };
    Json(openai::to_chat_completion("chatcmpl-0", resp))
}

pub async fn openai_completion_handler(Json(req): Json<CompletionRequest>) -> Json<openai::Completion> {
    let model = req.model.clone();
    let now = OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_default();
    let resp = GenerateResponse {
        model,
        created_at: now,
        response: "not implemented".into(),
        done: true,
        ..Default::default()
    };
    Json(openai::to_completion("cmpl-0", resp))
}

pub async fn openai_embeddings_handler(Json(req): Json<EmbedRequest>) -> Json<openai::EmbeddingList> {
    let resp = EmbedResponse {
        model: req.model.clone(),
        embeddings: vec![vec![0.0]],
        total_duration: None,
        load_duration: None,
        prompt_eval_count: Some(0),
    };
    Json(openai::to_embedding_list(&req.model, resp))
}

pub async fn openai_models_handler(State(state): State<Arc<AppState>>) -> Json<openai::ListCompletion> {
    let models = state.models.lock().unwrap();
    let resp = ListResponse { models: models.values().cloned().collect() };
    Json(openai::to_list_completion(resp))
}

pub async fn openai_model_handler(State(state): State<Arc<AppState>>, Path(model): Path<String>) -> Result<Json<openai::Model>, StatusCode> {
    let models = state.models.lock().unwrap();
    if let Some(m) = models.get(&model) {
        let resp = ShowResponse { modified_at: Some(m.modified_at.clone()), ..Default::default() };
        Ok(Json(openai::to_model(resp, &model)))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn auth_middleware(req: AxumRequest, next: Next) -> Result<Response, StatusCode> {
    if let Some(header) = req.headers().get(axum::http::header::AUTHORIZATION) {
        if let Ok(auth_str) = header.to_str() {
            let msg = format!("{}:{}", req.method(), req.uri().path());
            if verify_signature(auth_str, msg.as_bytes()) {
                return Ok(next.run(req).await);
            }
        }
    }
    Err(StatusCode::UNAUTHORIZED)
}

fn verify_signature(header: &str, message: &[u8]) -> bool {
    let mut parts = header.split(':');
    let key_b64 = match parts.next() { Some(s) => s, None => return false };
    let sig_b64 = match parts.next() { Some(s) => s, None => return false };
    let key_bytes = match STANDARD.decode(key_b64) { Ok(b) => b, Err(_) => return false };
    let sig_bytes = match STANDARD.decode(sig_b64) { Ok(b) => b, Err(_) => return false };
    let key_array: [u8; 32] = match key_bytes.try_into() { Ok(a) => a, Err(_) => return false };
    let sig_array: [u8; 64] = match sig_bytes.try_into() { Ok(a) => a, Err(_) => return false };
    let verifying = match VerifyingKey::from_bytes(&key_array) { Ok(v) => v, Err(_) => return false };
    let signature = Signature::from_bytes(&sig_array);
    verifying.verify(message, &signature).is_ok()
}

