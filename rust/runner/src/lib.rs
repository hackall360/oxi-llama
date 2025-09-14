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
    /// Generate embeddings for an image. Implementations may return one or
    /// more embeddings depending on the underlying model.
    fn embed_image(&mut self, data: &[u8]) -> Result<Vec<Vec<f32>>, Box<dyn Error>>;

    /// Determine the batch size to use for embedding processing. The default
    /// implementation simply returns the configured batch size which mirrors
    /// the behaviour of the Go runner.
    fn batch_size(&self, configured_batch_size: usize) -> usize {
        configured_batch_size
    }

    /// Size of a single embedding vector. Not all tests require this value so
    /// a default implementation returning `0` is provided.
    fn embed_size(&self) -> usize {
        0
    }
}
