use model::{
    default_config_path, get_text_processor, parse_tags, register_model, Model, ModelError,
    RegisteredModel, Special, TextProcessor, Vocabulary,
};
use once_cell::sync::Lazy;
use std::env;

struct DummyTP;
impl Model for DummyTP {}
impl TextProcessor for DummyTP {
    fn encode(&self, _s: &str, _add_special: bool) -> Result<Vec<i32>, anyhow::Error> {
        Ok(vec![])
    }
    fn decode(&self, _ids: &[i32]) -> Result<String, anyhow::Error> {
        Ok(String::new())
    }
    fn is(&self, _id: i32, _special: Special) -> bool {
        false
    }
    fn vocabulary(&self) -> &Vocabulary {
        static V: Lazy<Vocabulary> = Lazy::new(|| Vocabulary::new());
        &V
    }
}

struct NotTextProcessor;
impl Model for NotTextProcessor {}

#[test]
fn test_parse_tags() {
    let t = parse_tags("output");
    assert_eq!(t.name, "output");
    assert!(t.alternate.is_empty());
    let t = parse_tags("output,alt:token_embd");
    assert_eq!(t.name, "output");
    assert_eq!(t.alternate, vec!["token_embd"]);
}

#[test]
fn test_get_text_processor() {
    match get_text_processor("dummy") {
        Err(ModelError::Unsupported(_)) => {}
        _ => panic!("unexpected error"),
    }
    register_model("dummy", || {
        RegisteredModel::Other(Box::new(NotTextProcessor))
    });
    match get_text_processor("dummy") {
        Err(ModelError::NotTextProcessor) => {}
        _ => panic!("unexpected error"),
    }
}

#[test]
fn test_default_config_path_env() {
    let dir = tempfile::tempdir().unwrap();
    env::set_var("OLLAMA_CONFIG", dir.path());
    assert_eq!(default_config_path(), dir.path());
    env::remove_var("OLLAMA_CONFIG");
}
