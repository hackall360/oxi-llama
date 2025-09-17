use std::env;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use parking_lot::Mutex;
use reqwest::blocking::Client;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use super::paths::Paths;

#[derive(Debug)]
pub struct ServerHandle {
    command_tx: Sender<ServerCommand>,
    done_rx: Receiver<i32>,
    join_handle: Option<thread::JoinHandle<()>>,
}

impl ServerHandle {
    pub fn shutdown(&mut self) -> Result<i32> {
        let _ = self.command_tx.send(ServerCommand::Shutdown);
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
        Ok(self.done_rx.recv().unwrap_or(0))
    }
}

enum ServerCommand {
    Shutdown,
}

enum WaitOutcome {
    Shutdown(i32),
    Crash(i32),
}

pub fn spawn(paths: &Paths) -> Result<ServerHandle> {
    let cli_path = resolve_cli_path(paths);
    let paths_clone = paths.clone();
    let (command_tx, command_rx) = bounded::<ServerCommand>(1);
    let (done_tx, done_rx) = bounded::<i32>(1);

    let join_handle = thread::Builder::new()
        .name("ollama-server".into())
        .spawn(move || server_loop(paths_clone, cli_path, command_rx, done_tx))?;

    Ok(ServerHandle {
        command_tx,
        done_rx,
        join_handle: Some(join_handle),
    })
}

fn server_loop(
    paths: Paths,
    cli_path: PathBuf,
    command_rx: Receiver<ServerCommand>,
    done_tx: Sender<i32>,
) {
    let mut crash_count = 0;
    loop {
        if let Ok(ServerCommand::Shutdown) = command_rx.try_recv() {
            let _ = done_tx.send(0);
            break;
        }

        match start_server(&paths, &cli_path) {
            Ok(mut child) => match wait_for_child(&mut child, &command_rx) {
                Ok(WaitOutcome::Shutdown(code)) => {
                    let _ = done_tx.send(code);
                    break;
                }
                Ok(WaitOutcome::Crash(code)) => {
                    crash_count += 1;
                    warn!(
                        code,
                        crash_count, "ollama server exited unexpectedly, respawning"
                    );
                    backoff(crash_count);
                }
                Err(err) => {
                    crash_count += 1;
                    error!(?err, "error waiting for server");
                    backoff(crash_count);
                }
            },
            Err(err) => {
                crash_count += 1;
                error!(?err, "failed to start ollama server");
                backoff(crash_count);
            }
        }
    }
}

fn backoff(attempt: usize) {
    let delay =
        Duration::from_millis((attempt as u64).saturating_mul(500)).min(Duration::from_secs(5));
    thread::sleep(delay);
}

fn start_server(paths: &Paths, cli_path: &Path) -> Result<Child> {
    rotate_log(paths.server_log_file(), paths.log_rotation_count())?;
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(paths.server_log_file())
        .with_context(|| {
            format!(
                "failed to open server log {}",
                paths.server_log_file().display()
            )
        })?;
    let log_shared = Mutex::new(log_file);
    let log_arc = Arc::new(log_shared);

    let mut command = Command::new(cli_path);
    command.arg("serve");
    command.current_dir(paths.app_dir());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        command.creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP);
    }

    debug!(command = ?command, "starting ollama server process");
    let mut child = command.spawn().context("failed to spawn ollama server")?;

    if let Some(stdout) = child.stdout.take() {
        let log = log_arc.clone();
        thread::spawn(move || copy_stream(BufReader::new(stdout), log, "stdout"));
    }
    if let Some(stderr) = child.stderr.take() {
        let log = log_arc.clone();
        thread::spawn(move || copy_stream(BufReader::new(stderr), log, "stderr"));
    }

    info!(pid = child.id(), "started ollama server");

    Ok(child)
}

fn copy_stream<R>(mut reader: BufReader<R>, log: Arc<Mutex<fs::File>>, label: &str)
where
    R: Read + Send + 'static,
{
    let mut buffer = Vec::with_capacity(4096);
    loop {
        buffer.clear();
        match reader.read_until(b'\n', &mut buffer) {
            Ok(0) => break,
            Ok(_) => {
                let mut file = log.lock();
                if let Err(err) = file.write_all(&buffer) {
                    debug!(?err, stream = label, "failed to write to server log");
                    break;
                }
            }
            Err(err) => {
                debug!(?err, stream = label, "error reading server stream");
                break;
            }
        }
    }
}

fn wait_for_child(child: &mut Child, commands: &Receiver<ServerCommand>) -> Result<WaitOutcome> {
    loop {
        match commands.recv_timeout(Duration::from_millis(200)) {
            Ok(ServerCommand::Shutdown) => {
                debug!("shutting down ollama server");
                graceful_terminate(child)?;
                let status = child.wait()?;
                let code = status.code().unwrap_or_default();
                return Ok(WaitOutcome::Shutdown(code));
            }
            Err(RecvTimeoutError::Timeout) => match child.try_wait() {
                Ok(Some(status)) => {
                    let code = status.code().unwrap_or_default();
                    return Ok(WaitOutcome::Crash(code));
                }
                Ok(None) => continue,
                Err(err) => return Err(err.into()),
            },
            Err(RecvTimeoutError::Disconnected) => {
                graceful_terminate(child)?;
                let status = child.wait()?;
                let code = status.code().unwrap_or_default();
                return Ok(WaitOutcome::Shutdown(code));
            }
        }
    }
}

fn graceful_terminate(child: &mut Child) -> Result<()> {
    send_terminate_signal(child)?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return Ok(()),
            Ok(None) => {
                if start.elapsed() > Duration::from_secs(5) {
                    warn!(pid = child.id(), "server did not exit gracefully, killing");
                    child.kill()?;
                    return Ok(());
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(err) => return Err(err.into()),
        }
    }
}

#[cfg(unix)]
fn send_terminate_signal(child: &Child) -> Result<()> {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;

    let pid = child.id() as i32;
    kill(Pid::from_raw(pid), Signal::SIGINT).map_err(|err| anyhow!(err))
}

#[cfg(windows)]
fn send_terminate_signal(child: &Child) -> Result<()> {
    use windows::Win32::Foundation::BOOL;
    use windows::Win32::System::Console::{
        AttachConsole, FreeConsole, GenerateConsoleCtrlEvent, SetConsoleCtrlHandler,
        CTRL_BREAK_EVENT, CTRL_C_EVENT,
    };

    let pid = child.id();
    unsafe {
        let _ = AttachConsole(pid);
        SetConsoleCtrlHandler(None, true).map_err(|err| anyhow!(err.message()))?;
        GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid).map_err(|err| anyhow!(err.message()))?;
        GenerateConsoleCtrlEvent(CTRL_C_EVENT, pid).map_err(|err| anyhow!(err.message()))?;
        let _ = FreeConsole();
    }
    Ok(())
}

fn rotate_log(path: &Path, count: usize) -> Result<()> {
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
        .unwrap_or_else(|| String::from("log"));
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
            fs::remove_file(&dst)?;
        }
        fs::rename(&src, &dst)?;
    }

    Ok(())
}

fn build_rotated_path(base: &Path, stem: &str, ext: &str, index: usize) -> PathBuf {
    let parent = base.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!("{}-{}{}", stem, index, ext))
}

fn resolve_cli_path(paths: &Paths) -> PathBuf {
    let mut candidates = Vec::new();
    let command = paths.cli_name();
    candidates.push(paths.app_dir().join(command));
    candidates.push(paths.app_dir().join("bin").join(command));
    if let Ok(found) = which::which(command) {
        candidates.push(found);
    }
    if let Ok(current_dir) = env::current_dir() {
        candidates.push(current_dir.join(command));
    }

    for candidate in candidates {
        if candidate.exists() {
            return candidate;
        }
    }

    PathBuf::from(command)
}

pub fn is_server_running() -> bool {
    let host = envconfig::host();
    let url = if host.ends_with('/') {
        host
    } else {
        format!("{}/", host)
    };
    let client = Client::builder().timeout(Duration::from_secs(2)).build();
    match client {
        Ok(client) => match client.head(&url).send() {
            Ok(resp) => resp.status().is_success(),
            Err(err) => {
                debug!(?err, "heartbeat request failed");
                false
            }
        },
        Err(err) => {
            debug!(?err, "failed to build http client");
            false
        }
    }
}
