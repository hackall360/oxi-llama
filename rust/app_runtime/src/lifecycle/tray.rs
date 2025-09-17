use anyhow::Result;
use crossbeam_channel::{unbounded, Receiver};
use tracing::warn;

#[derive(Debug, Clone)]
pub enum TrayEvent {
    Quit,
    InstallUpdate,
    ShowLogs,
    FirstUse,
}

#[derive(Clone)]
pub struct TrayController {
    events: Receiver<TrayEvent>,
}

impl TrayController {
    pub fn new() -> Result<Self> {
        #[cfg(windows)]
        {
            return crate::lifecycle::tray_windows::create_tray();
        }

        #[cfg(not(windows))]
        {
            warn!("system tray not available on this platform");
            let (_tx, rx) = unbounded();
            Ok(Self { events: rx })
        }
    }

    pub fn events(&self) -> Receiver<TrayEvent> {
        self.events.clone()
    }

    pub fn notify_update(&self, _version: &str) -> Result<()> {
        #[cfg(not(windows))]
        {
            warn!("update notification unavailable without tray support");
            return Ok(());
        }
        #[cfg(windows)]
        {
            crate::lifecycle::tray_windows::notify_update(_version)
        }
    }

    pub fn notify_first_use(&self) -> Result<()> {
        #[cfg(not(windows))]
        {
            warn!("first use notification unavailable without tray support");
            return Ok(());
        }
        #[cfg(windows)]
        {
            crate::lifecycle::tray_windows::notify_first_use()
        }
    }

    pub fn request_exit(&self) -> Result<()> {
        #[cfg(not(windows))]
        {
            Ok(())
        }
        #[cfg(windows)]
        {
            crate::lifecycle::tray_windows::request_exit()
        }
    }

    pub fn join(&self) {}
}

#[cfg(windows)]
mod tray_windows {
    use super::TrayEvent;
    use anyhow::Result;
    use crossbeam_channel::{unbounded, Receiver, Sender};
    use std::sync::OnceLock;
    use tracing::{info, warn};

    static EVENTS: OnceLock<(Sender<TrayEvent>, Receiver<TrayEvent>)> = OnceLock::new();

    pub fn create_tray() -> Result<super::TrayController> {
        let (tx, rx) = EVENTS.get_or_init(|| unbounded()).clone();
        info!("tray integration for windows is not yet implemented");
        Ok(super::TrayController { events: rx })
    }

    pub fn notify_update(_version: &str) -> Result<()> {
        warn!("windows tray integration not implemented");
        Ok(())
    }

    pub fn notify_first_use() -> Result<()> {
        warn!("windows tray integration not implemented");
        Ok(())
    }

    pub fn request_exit() -> Result<()> {
        warn!("windows tray integration not implemented");
        Ok(())
    }
}
