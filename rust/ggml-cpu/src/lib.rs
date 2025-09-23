//! Rust-native CPU kernels for GGML primitives.
//! This module reimplements selected parts of the original C++ backend
//! using portable SIMD abstractions provided by [`packed_simd_2`].

mod arch;
mod binary;
mod detection;
mod simd_common;
mod unary;
mod vec;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use arch::amx::matmul_i8_tile;
pub use binary::*;
pub use detection::{capabilities, CpuCapabilities};
pub use unary::*;
pub use vec::*;
