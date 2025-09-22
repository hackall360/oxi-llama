use std::fmt;

use thiserror::Error;

use crate::tensor::TensorId;

#[derive(Debug, Error)]
pub enum Error {
    #[error(
        "out of memory: requested {requested} bytes but only {available} bytes remain in arena"
    )]
    OutOfMemory { requested: usize, available: usize },
    #[error("invalid alignment: {0}")]
    InvalidAlignment(usize),
    #[error("invalid shape: {0:?}")]
    InvalidShape(Vec<usize>),
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("dtype mismatch: {0}")]
    DTypeMismatch(String),
    #[error("graph contains a cycle")]
    GraphCycle,
    #[error("tensor {0:?} has not been computed")]
    UninitializedTensor(TensorId),
    #[error("tensor belongs to a different context")]
    ContextMismatch,
    #[error("operation error: {0}")]
    Operation(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn dtype_mismatch(expected: impl fmt::Display, actual: impl fmt::Display) -> Self {
        Self::DTypeMismatch(format!("expected {expected}, got {actual}"))
    }
}
