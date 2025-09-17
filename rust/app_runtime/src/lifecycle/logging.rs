use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result};
use atty::Stream;
use tracing::Level;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::EnvFilter;

use super::paths::Paths;

static INIT_ONCE: OnceLock<()> = OnceLock::new();

pub struct LoggingGuard {
    _guard: Option<WorkerGuard>,
}

pub fn init(paths: &Paths) -> Result<LoggingGuard> {
    if INIT_ONCE.get().is_some() {
        return Ok(LoggingGuard { _guard: None });
    }

    if atty::is(Stream::Stderr) {
        logutil::init();
        tracing::info!("ollama app started");
        INIT_ONCE.set(()).ok();
        return Ok(LoggingGuard { _guard: None });
    }

    rotate_logs(paths.app_log_file(), paths.log_rotation_count())?;

    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(paths.app_log_file())
        .with_context(|| format!("failed to open log file {}", paths.app_log_file().display()))?;
    let (writer, guard) = tracing_appender::non_blocking(log_file);

    let default_level = std::env::var("OLLAMA_LOG_LEVEL")
        .ok()
        .and_then(|lvl| logutil::parse_level(&lvl))
        .unwrap_or(Level::INFO);

    let filter = EnvFilter::builder()
        .with_default_directive(default_level.into())
        .from_env_lossy();

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(writer)
        .with_ansi(false)
        .init();

    tracing::info!("ollama app started");
    INIT_ONCE.set(()).ok();

    Ok(LoggingGuard {
        _guard: Some(guard),
    })
}

fn rotate_logs(path: &Path, count: usize) -> Result<()> {
    if count == 0 {
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create log directory {}", parent.display()))?;
    }

    if !path.exists() {
        return Ok(());
    }

    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| String::from("app"));
    let ext = path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();

    for idx in (1..=count).rev() {
        let src = if idx == 1 {
            path.to_path_buf()
        } else {
            build_rotated_path(path, &stem, &ext, idx - 1)
        };
        if !src.exists() {
            continue;
        }

        let dst = build_rotated_path(path, &stem, &ext, idx);
        if dst.exists() {
            fs::remove_file(&dst)
                .with_context(|| format!("failed to remove old log {}", dst.display()))?;
        }

        fs::rename(&src, &dst).with_context(|| {
            format!(
                "failed to rotate log {} -> {}",
                src.display(),
                dst.display()
            )
        })?;
    }

    Ok(())
}

fn build_rotated_path(base: &Path, stem: &str, ext: &str, index: usize) -> PathBuf {
    let parent = base.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!("{}-{}{}", stem, index, ext))
}
