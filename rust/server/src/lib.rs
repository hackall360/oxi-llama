use axum::{routing::{post, delete, get}, Router, Json, extract::{State, Path}};
use std::sync::Arc;
use serde_json::{json, Value};
use api::{CreateRequest, GenerateRequest, DeleteRequest, GenerateResponse, ChatResponse, EmbedResponse, ListResponse, ShowResponse, Message as ApiMessage};
use api::openai::{self, ChatCompletionRequest, CompletionRequest, EmbedRequest};
use time::OffsetDateTime;
use convert::{convert_model, ModelFormat};
use std::path::Path as StdPath;
#[derive(Clone, Default)]
pub struct AppState {
    // placeholder for shared state like DB, model, etc.
}

pub fn app() -> Router<Arc<AppState>> {
    let state = Arc::new(AppState::default());
    Router::new()
        .route("/api/create", post(create_handler))
        .route("/api/generate", post(generate_handler))
        .route("/api/delete", delete(delete_handler))
        .route("/v1/chat/completions", post(openai_chat_handler))
        .route("/v1/completions", post(openai_completion_handler))
        .route("/v1/embeddings", post(openai_embeddings_handler))
        .route("/v1/models", get(openai_models_handler))
        .route("/v1/models/:model", get(openai_model_handler))
        .with_state(state)
}

pub async fn create_handler(State(_state): State<Arc<AppState>>, Json(req): Json<CreateRequest>) -> Json<Value> {
    let _ = convert_model(StdPath::new(&req.model), StdPath::new("model.bin"), ModelFormat::LLaMA);
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

pub async fn delete_handler(State(_state): State<Arc<AppState>>, Json(_req): Json<DeleteRequest>) -> Json<Value> {
    Json(json!({"status": "ok"}))
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

pub async fn openai_models_handler() -> Json<openai::ListCompletion> {
    let resp = ListResponse { models: vec![] };
    Json(openai::to_list_completion(resp))
}

pub async fn openai_model_handler(Path(model): Path<String>) -> Json<openai::Model> {
    let resp = ShowResponse { modified_at: Some(OffsetDateTime::now_utc().format(&time::format_description::well_known::Rfc3339).unwrap()), ..Default::default() };
    Json(openai::to_model(resp, &model))
}

