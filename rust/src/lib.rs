pub use envconfig;

pub mod format;
pub use format::{format_bytes, format_time};

extern "C" {
    fn hello_from_c() -> i32;
}

pub fn call_c() -> i32 {
    unsafe { hello_from_c() }
}
