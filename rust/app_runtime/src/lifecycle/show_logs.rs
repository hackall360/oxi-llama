use anyhow::Result;

use super::paths::Paths;

pub fn open_logs(paths: &Paths) -> Result<()> {
    open::that(paths.app_data_dir())?;
    Ok(())
}
