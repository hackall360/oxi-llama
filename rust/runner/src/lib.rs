pub mod common;
pub mod llamarunner;
pub mod ollamarunner;

use std::error::Error;

/// Trait for loading models into the runner backend.
pub trait ModelLoader {
    fn load_model(&mut self, path: &str) -> Result<(), Box<dyn Error>>;
}

/// Trait representing a cache backend for KV data.
pub trait KvCache {
    fn remove(&mut self, slot: usize, begin: i32, end: i32) -> Result<(), Box<dyn Error>>;
    fn copy_prefix(&mut self, src: usize, dst: usize, len: i32);
    fn can_shift(&self) -> bool {
        false
    }
    fn shift(&mut self, _slot: usize, _start: i32, _end: i32, _delta: i32) {}
}

/// Trait representing multimodal processing such as vision features.
pub trait MultimodalSupport {
    fn embed_image(&self, data: &[u8]) -> Vec<f32>;
}
