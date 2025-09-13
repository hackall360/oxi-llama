use server::{AppState, create_handler, generate_handler, delete_handler};
use api::{CreateRequest, GenerateRequest, DeleteRequest};
use axum::{Json, extract::State};
use std::sync::Arc;

#[tokio::test]
async fn test_create_handler() {
    let state = Arc::new(AppState::default());
    let req = CreateRequest { model: "test".into(), ..Default::default() };
    let Json(resp) = create_handler(State(state), Json(req)).await;
    assert_eq!(resp["status"], "ok");
}

#[tokio::test]
async fn test_generate_handler() {
    let state = Arc::new(AppState::default());
    let req = GenerateRequest { model: "test".into(), ..Default::default() };
    let Json(resp) = generate_handler(State(state), Json(req)).await;
    assert!(resp.done);
}

#[tokio::test]
async fn test_delete_handler() {
    let state = Arc::new(AppState::default());
    let req = DeleteRequest { model: "test".into(), ..Default::default() };
    let Json(resp) = delete_handler(State(state), Json(req)).await;
    assert_eq!(resp["status"], "ok");
}
