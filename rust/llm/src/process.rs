use std::env;
use std::process::{Child, Command};

/// Returns environment variables filtered to only those that should be passed to the runner.
pub fn filtered_env() -> Vec<(String, String)> {
    let allow_keys = ["PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"];
    let mut envs = Vec::new();
    for (key, value) in env::vars() {
        let upper = key.to_ascii_uppercase();
        let allow = upper.starts_with("OLLAMA_")
            || upper.starts_with("CUDA_")
            || upper.starts_with("ROCR_")
            || upper.starts_with("ROCM_")
            || upper.starts_with("HIP_")
            || upper.starts_with("GPU_")
            || upper.starts_with("HSA_")
            || upper.starts_with("GGML_")
            || allow_keys.iter().any(|k| k.eq(&upper));
        if allow {
            envs.push((key, value));
        }
    }
    envs
}

/// Spawn a new process for the llama runner using filtered environment variables.
pub fn spawn_llama_server(cmd_path: &str, args: &[&str]) -> std::io::Result<Child> {
    let mut cmd = Command::new(cmd_path);
    cmd.args(args);
    for (k, v) in filtered_env() {
        cmd.env(k, v);
    }
    apply_platform_defaults(&mut cmd);
    cmd.spawn()
}

#[cfg(target_os = "windows")]
fn apply_platform_defaults(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_DEFAULT_ERROR_MODE: u32 = 0x04000000;
    const ABOVE_NORMAL_PRIORITY_CLASS: u32 = 0x00008000;
    const CREATE_NO_WINDOW: u32 = 0x08000000;
    cmd.creation_flags(CREATE_DEFAULT_ERROR_MODE | ABOVE_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW);
}

#[cfg(not(target_os = "windows"))]
fn apply_platform_defaults(_cmd: &mut Command) {}
