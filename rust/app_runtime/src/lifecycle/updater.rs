use std::thread;
use std::time::Duration;

use anyhow::Result;
use tracing::{debug, info, warn};

use super::tray::TrayController;

const UPDATE_CHECK_INTERVAL: Duration = Duration::from_secs(60 * 60);

pub fn start_background_updater_checker(tray: TrayController) {
    thread::spawn(move || loop {
        debug!("background updater check starting");
        if let Err(err) = tray.notify_update("pending") {
            warn!(?err, "unable to notify update availability");
        }
        thread::sleep(UPDATE_CHECK_INTERVAL);
    });
}

pub fn download_new_release() -> Result<()> {
    info!("download_new_release stub - no operation performed");
    Ok(())
}

pub fn do_upgrade() -> Result<()> {
    info!("do_upgrade stub - no operation performed");
    Ok(())
}
