use std::env;
use std::path::PathBuf;

/// Return the directory for configuration files.
///
/// If the `OLLAMA_CONFIG` environment variable is set, it is used directly.
/// Otherwise the directory `~/.ollama` is returned (falling back to the
/// current directory if the home directory is unknown).
pub fn config_dir() -> PathBuf {
    if let Some(val) = env::var_os("OLLAMA_CONFIG") {
        PathBuf::from(val)
    } else if let Some(home) = env::var_os("HOME") {
        PathBuf::from(home).join(".ollama")
    } else {
        PathBuf::from(".")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_override() {
        let dir = env::temp_dir();
        env::set_var("OLLAMA_CONFIG", dir.to_str().unwrap());
        assert_eq!(config_dir(), dir);
        env::remove_var("OLLAMA_CONFIG");
    }

    #[test]
    fn home_default() {
        env::remove_var("OLLAMA_CONFIG");
        let dir = tempfile::tempdir().unwrap();
        env::set_var("HOME", dir.path());
        assert_eq!(config_dir(), dir.path().join(".ollama"));
        env::remove_var("HOME");
    }
}
