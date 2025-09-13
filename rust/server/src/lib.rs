use axum::{routing::{post, delete}, Router, Json, extract::State};
use std::sync::Arc;
use serde_json::{json, Value};
use api::{CreateRequest, GenerateRequest, DeleteRequest, GenerateResponse};
use convert::{convert_model, ModelFormat};
use std::path::Path;
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
        .with_state(state)
}

pub async fn create_handler(State(_state): State<Arc<AppState>>, Json(req): Json<CreateRequest>) -> Json<Value> {
    let _ = convert_model(Path::new(&req.model), Path::new("model.bin"), ModelFormat::LLaMA);
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

