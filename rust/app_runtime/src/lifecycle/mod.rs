use anyhow::Result;
use crossbeam_channel::{select, unbounded};
use tracing::{info, warn};

mod get_started;
pub mod logging;
pub mod paths;
mod server;
mod show_logs;
mod store;
mod tray;
mod updater;

pub fn run() -> Result<()> {
    let paths = paths::initialize()?;
    let _logging_guard = logging::init(paths)?;
    info!(host = %envconfig::host(), "app configuration loaded");

    let store = store::Store::open(paths)?;
    let tray = tray::TrayController::new()?;
    let tray_events = tray.events();

    let (signal_tx, signal_rx) = unbounded();
    ctrlc::set_handler(move || {
        let _ = signal_tx.send(());
    })?;

    if !store.first_time_run() {
        info!("first time run detected");
        tray.notify_first_use().ok();
        store.set_first_time_run(true)?;
    }

    if server::is_server_running() {
        warn!("detected running server instance");
    }

    let mut server = match server::spawn(paths) {
        Ok(handle) => Some(handle),
        Err(err) => {
            warn!(?err, "failed to launch server");
            None
        }
    };

    updater::start_background_updater_checker(tray.clone());

    loop {
        select! {
            recv(tray_events) -> event => {
                match event {
                    Ok(tray::TrayEvent::Quit) => {
                        info!("quit requested from tray");
                        break;
                    }
                    Ok(tray::TrayEvent::ShowLogs) => {
                        if let Err(err) = show_logs::open_logs(paths) {
                            warn!(?err, "failed to open logs");
                        }
                    }
                    Ok(tray::TrayEvent::FirstUse) => {
                        if let Err(err) = get_started::launch(paths) {
                            warn!(?err, "failed to open getting started guide");
                        }
                    }
                    Ok(tray::TrayEvent::InstallUpdate) => {
                        if let Err(err) = updater::do_upgrade() {
                            warn!(?err, "upgrade failed");
                        }
                    }
                    Err(_) => {}
                }
            }
            recv(signal_rx) -> _ => {
                info!("shutdown signal received");
                break;
            }
        }
    }

    tray.request_exit().ok();
    if let Some(ref mut handle) = server {
        if let Err(err) = handle.shutdown() {
            warn!(?err, "failed to shutdown server cleanly");
        }
    }
    tray.join();
    info!("lifecycle shutdown complete");
    Ok(())
}
