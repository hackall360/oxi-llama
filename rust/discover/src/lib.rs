pub mod common;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod darwin;
#[cfg(target_os = "windows")]
mod windows;

pub use common::*;

#[cfg(target_os = "linux")]
pub use linux::*;
#[cfg(target_os = "macos")]
pub use darwin::*;
#[cfg(target_os = "windows")]
pub use windows::*;
