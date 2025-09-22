pub use envconfig;
pub use logutil::{self, debug, error, info, trace, warn, Level};

pub mod format;
pub use format::{format_bytes, format_time};

/// Returns a canned value that previously came from the C helper. Now
/// implemented purely in Rust.
pub fn helper_value() -> i32 {
    42
}
