use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use once_cell::sync::OnceCell;

#[derive(Clone, Debug)]
pub struct Paths {
    app_name: String,
    cli_name: String,
    app_dir: PathBuf,
    app_data_dir: PathBuf,
    update_stage_dir: PathBuf,
    app_log_file: PathBuf,
    server_log_file: PathBuf,
    upgrade_log_file: PathBuf,
    installer: String,
    log_rotation_count: usize,
}

static GLOBAL_PATHS: OnceCell<Paths> = OnceCell::new();

pub fn initialize() -> Result<&'static Paths> {
    GLOBAL_PATHS.get_or_try_init(Paths::detect)
}

pub fn current() -> &'static Paths {
    GLOBAL_PATHS.get().expect("paths not initialized")
}

impl Paths {
    fn detect() -> Result<Paths> {
        let mut app_name = String::from("ollama app");
        let mut cli_name = String::from("ollama");
        let mut installer = String::from("OllamaInstaller");
        let log_rotation_count = 5usize;

        let default_app_dir = env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let mut app_dir = default_app_dir.clone();
        let mut app_data_dir: PathBuf;
        let mut update_stage_dir: PathBuf;
        let mut app_log_file: PathBuf;
        let mut server_log_file: PathBuf;
        let mut upgrade_log_file: PathBuf;

        #[cfg(windows)]
        {
            use std::ffi::OsString;

            app_name.push_str(".exe");
            cli_name.push_str(".exe");
            installer = String::from("OllamaSetup.exe");

            let local_app_data = env::var("LOCALAPPDATA")
                .map(PathBuf::from)
                .context("LOCALAPPDATA not set")?;
            app_data_dir = local_app_data.join("Ollama");
            update_stage_dir = app_data_dir.join("updates");
            app_log_file = app_data_dir.join("app.log");
            server_log_file = app_data_dir.join("server.log");
            upgrade_log_file = app_data_dir.join("upgrade.log");

            if let Ok(exe) = env::current_exe() {
                if let Some(dir) = exe.parent() {
                    app_dir = dir.to_path_buf();
                }
            }

            ensure_dir(&app_data_dir)?;

            let existing_paths: Vec<PathBuf> = env::var_os("PATH")
                .map(|paths| env::split_paths(&paths).collect())
                .unwrap_or_default();
            let mut paths = existing_paths.clone();
            if !existing_paths
                .iter()
                .any(|p| normalize_case(p) == normalize_case(&app_dir))
            {
                paths.push(app_dir.clone());
                let new_path = env::join_paths(paths).context("failed to update PATH")?;
                env::set_var("PATH", new_path);
            }
        }

        #[cfg(target_os = "macos")]
        {
            app_name.push_str(".app");
            installer = String::from("OllamaInstaller.pkg");

            let home = env::var("HOME").context("HOME not set")?;
            app_data_dir = PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("Ollama");
            update_stage_dir = app_data_dir.join("updates");
            app_log_file = app_data_dir.join("app.log");
            server_log_file = app_data_dir.join("server.log");
            upgrade_log_file = app_data_dir.join("upgrade.log");

            if let Ok(exe) = env::current_exe() {
                if let Some(dir) = exe.parent() {
                    app_dir = dir.to_path_buf();
                }
            }

            ensure_dir(&app_data_dir)?;
        }

        #[cfg(all(unix, not(target_os = "macos"), not(target_os = "android")))]
        {
            use nix::unistd::Uid;
            let is_root = Uid::effective().is_root();
            if is_root {
                app_data_dir = PathBuf::from("/var/lib/ollama");
                update_stage_dir = app_data_dir.join("updates");
                app_log_file = PathBuf::from("/var/log/ollama/app.log");
                server_log_file = PathBuf::from("/var/log/ollama/server.log");
                upgrade_log_file = PathBuf::from("/var/log/ollama/upgrade.log");
            } else {
                let home = env::var("HOME").context("HOME not set")?;
                app_data_dir = PathBuf::from(home).join(".ollama");
                update_stage_dir = app_data_dir.join("updates");
                let log_dir = app_data_dir.clone();
                app_log_file = log_dir.join("app.log");
                server_log_file = log_dir.join("server.log");
                upgrade_log_file = log_dir.join("upgrade.log");
            }

            if let Ok(exe) = env::current_exe() {
                if let Some(dir) = exe.parent() {
                    app_dir = dir.to_path_buf();
                }
            }

            ensure_dir(&app_data_dir)?;
            if let Some(parent) = app_log_file.parent() {
                ensure_dir(parent)?;
            }
            if let Some(parent) = server_log_file.parent() {
                ensure_dir(parent)?;
            }
            if let Some(parent) = upgrade_log_file.parent() {
                ensure_dir(parent)?;
            }
        }

        #[cfg(not(any(
            windows,
            target_os = "macos",
            all(unix, not(target_os = "macos"), not(target_os = "android"))
        )))]
        {
            app_data_dir = default_app_dir.clone();
            update_stage_dir = env::temp_dir().join("ollama-updates");
            app_log_file = env::temp_dir().join("ollama_app.log");
            server_log_file = env::temp_dir().join("ollama.log");
            upgrade_log_file = env::temp_dir().join("ollama_update.log");
        }

        Ok(Paths {
            app_name,
            cli_name,
            app_dir,
            app_data_dir,
            update_stage_dir,
            app_log_file,
            server_log_file,
            upgrade_log_file,
            installer,
            log_rotation_count,
        })
    }

    pub fn app_name(&self) -> &str {
        &self.app_name
    }

    pub fn cli_name(&self) -> &str {
        &self.cli_name
    }

    pub fn app_dir(&self) -> &Path {
        &self.app_dir
    }

    pub fn app_data_dir(&self) -> &Path {
        &self.app_data_dir
    }

    pub fn update_stage_dir(&self) -> &Path {
        &self.update_stage_dir
    }

    pub fn app_log_file(&self) -> &Path {
        &self.app_log_file
    }

    pub fn server_log_file(&self) -> &Path {
        &self.server_log_file
    }

    pub fn upgrade_log_file(&self) -> &Path {
        &self.upgrade_log_file
    }

    pub fn installer(&self) -> &str {
        &self.installer
    }

    pub fn log_rotation_count(&self) -> usize {
        self.log_rotation_count
    }

    pub fn store_file(&self) -> Result<PathBuf> {
        #[cfg(windows)]
        {
            return Ok(self.app_data_dir.join("config.json"));
        }

        #[cfg(target_os = "macos")]
        {
            return Ok(self.app_data_dir.join("config.json"));
        }

        #[cfg(all(unix, not(target_os = "macos"), not(target_os = "android")))]
        {
            use nix::unistd::Uid;
            if Uid::effective().is_root() {
                return Ok(PathBuf::from("/etc/ollama/config.json"));
            }
            return Ok(self.app_data_dir.join("config.json"));
        }

        #[allow(unreachable_code)]
        Err(anyhow!("unsupported platform"))
    }
}

fn ensure_dir(dir: &Path) -> Result<()> {
    if !dir.exists() {
        fs::create_dir_all(dir)
            .with_context(|| format!("failed to create directory {}", dir.display()))?;
    }
    Ok(())
}

#[cfg(windows)]
fn normalize_case(path: &Path) -> String {
    path.to_string_lossy().to_ascii_lowercase()
}

#[cfg(not(windows))]
fn normalize_case(path: &Path) -> String {
    path.to_string_lossy().to_string()
}
