#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod amx;
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
pub mod arm;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;
