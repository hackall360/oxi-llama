use std::collections::HashMap;
use std::env;

use llm::filtered_env;

#[test]
fn test_filtered_env_case_insensitive() {
    // Save existing vars to restore later
    let orig_path = env::var("Path").ok();
    let orig_ollama = env::var("OLLAMA_Test").ok();
    let orig_other = env::var("UNRELATED_VAR").ok();

    env::set_var("Path", "abc");
    env::set_var("OLLAMA_Test", "123");
    env::set_var("UNRELATED_VAR", "xxx");

    let envs = filtered_env();
    let map: HashMap<_, _> = envs.into_iter().collect();

    assert_eq!(map.get("Path"), Some(&"abc".to_string()));
    assert_eq!(map.get("OLLAMA_Test"), Some(&"123".to_string()));
    assert!(map.get("UNRELATED_VAR").is_none());

    // Restore
    if let Some(v) = orig_path {
        env::set_var("Path", v);
    } else {
        env::remove_var("Path");
    }
    if let Some(v) = orig_ollama {
        env::set_var("OLLAMA_Test", v);
    } else {
        env::remove_var("OLLAMA_Test");
    }
    if let Some(v) = orig_other {
        env::set_var("UNRELATED_VAR", v);
    } else {
        env::remove_var("UNRELATED_VAR");
    }
}
