use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VERSION");
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    if let Some(git_dir) = locate_git_dir(Path::new(&manifest_dir)) {
        println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
        println!("cargo:rerun-if-changed={}", git_dir.join("index").display());
    }

    let version = env::var("VERSION").unwrap_or_else(|_| env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".into()));
    println!("cargo:rustc-env=OXI_VERSION={version}");

    let git_commit = git_commit(&manifest_dir).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=OXI_GIT_COMMIT={git_commit}");

    let git_dirty = git_dirty(&manifest_dir).unwrap_or(false);
    println!("cargo:rustc-env=OXI_GIT_DIRTY={git_dirty}");

    let build_timestamp = build_timestamp();
    println!("cargo:rustc-env=OXI_BUILD_TIMESTAMP={build_timestamp}");

    let target = env::var("TARGET").unwrap_or_else(|_| "unknown-target".into());
    println!("cargo:rustc-env=OXI_BUILD_TARGET={target}");

    let profile = env::var("PROFILE").unwrap_or_else(|_| "unknown".into());
    println!("cargo:rustc-env=OXI_BUILD_PROFILE={profile}");

    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustc_version = rustc_version(&rustc).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=OXI_RUSTC_VERSION={rustc_version}");
}

fn git_commit(manifest_dir: &str) -> Option<String> {
    run_git_command(manifest_dir, ["rev-parse", "HEAD"])
}

fn git_dirty(manifest_dir: &str) -> Option<bool> {
    let output = run_git_command(manifest_dir, ["status", "--porcelain"])?;
    Some(!output.trim().is_empty())
}

fn run_git_command(manifest_dir: &str, args: impl IntoIterator<Item = &'static str>) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(manifest_dir)
        .output()
        .ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

fn build_timestamp() -> String {
    if let Ok(epoch) = env::var("SOURCE_DATE_EPOCH") {
        if let Ok(parsed) = epoch.parse::<u64>() {
            return parsed.to_string();
        }
    }
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "0".into())
}

fn rustc_version(rustc: &str) -> Option<String> {
    let output = Command::new(rustc).arg("--version").output().ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

fn locate_git_dir(start: &Path) -> Option<PathBuf> {
    let mut current = Some(start);
    while let Some(dir) = current {
        let dot_git = dir.join(".git");
        if dot_git.exists() {
            if dot_git.is_dir() {
                return Some(dot_git);
            }
            if let Ok(contents) = fs::read_to_string(&dot_git) {
                if let Some(path) = contents.strip_prefix("gitdir:") {
                    let git_path = path.trim();
                    let path = Path::new(git_path);
                    return Some(if path.is_absolute() { path.to_path_buf() } else { dir.join(path) });
                }
            }
        }
        current = dir.parent();
    }
    None
}
