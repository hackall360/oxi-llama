use api::{CreateRequest, DeleteRequest, GenerateRequest};
use axum::{extract::State, Json};
use server::{
    create_handler, delete_handler, generate_handler, list_handler, version_handler, AppState,
};
use std::sync::Arc;
use version::Version;

#[tokio::test]
async fn test_create_and_list() {
    let state = Arc::new(AppState::default());
    let req = CreateRequest {
        model: "m1".into(),
        ..Default::default()
    };
    let Json(_resp) = create_handler(State(state.clone()), Json(req)).await;

    let Json(list) = list_handler(State(state)).await;
    assert_eq!(list.models.len(), 1);
    assert_eq!(list.models[0].name, "m1");
}

#[tokio::test]
async fn test_version_handler() {
    let Json(body) = version_handler().await;
    assert_eq!(body["version"], Version);
    assert!(body["git_commit"].is_string());
    assert!(body["build_timestamp"].is_string());
}

#[tokio::test]
async fn test_generate_handler() {
    let state = Arc::new(AppState::default());
    let req = GenerateRequest {
        model: "test".into(),
        ..Default::default()
    };
    let Json(resp) = generate_handler(State(state), Json(req)).await;
    assert!(resp.done);
}

#[tokio::test]
async fn test_delete_handler() {
    let state = Arc::new(AppState::default());
    let create = CreateRequest {
        model: "m2".into(),
        ..Default::default()
    };
    let _ = create_handler(State(state.clone()), Json(create)).await;
    let req = DeleteRequest {
        model: "m2".into(),
        ..Default::default()
    };
    let Json(resp) = delete_handler(State(state), Json(req)).await;
    assert_eq!(resp["status"], "ok");
}
