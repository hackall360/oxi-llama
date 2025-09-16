#[path = "common/mod.rs"]
mod common;

#[test]
fn test_library_model_list_contains_expected_entries() {
    let models = common::list_library_models();
    assert!(
        !models.is_empty(),
        "expected curated model list to be non-empty"
    );
    assert!(
        models.iter().any(|&m| m.contains("llama3.2")),
        "llama-based model missing from curated list"
    );
    assert!(
        models.iter().any(|&m| m.starts_with("gemma")),
        "gemma entry missing from curated list"
    );
}

#[test]
fn test_library_embedding_models_include_minilm() {
    let models = common::list_embedding_models();
    assert!(
        models.contains(&"all-minilm"),
        "all-minilm should be part of embedding test set"
    );
}
