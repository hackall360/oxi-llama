use anyhow::Result;

use super::paths::Paths;

#[cfg(windows)]
pub fn launch(paths: &Paths) -> Result<()> {
    use std::os::windows::process::CommandExt;
    use std::process::Command;

    const CREATE_NEW_CONSOLE: u32 = 0x0000_0010;

    let script = paths.app_dir().join("ollama_welcome.ps1");
    if !script.exists() {
        anyhow::bail!("getting started script missing: {}", script.display());
    }

    let mut command = Command::new("powershell");
    command
        .arg("-noexit")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-nologo")
        .arg("-file")
        .arg(script.to_string_lossy().to_string())
        .creation_flags(CREATE_NEW_CONSOLE);

    command
        .spawn()
        .context("failed to launch getting started shell")?;
    Ok(())
}

#[cfg(not(windows))]
pub fn launch(_paths: &Paths) -> Result<()> {
    anyhow::bail!("getting started experience is only available on Windows")
}
