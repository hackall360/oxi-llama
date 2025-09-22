/// GGML tensor element types supported by the pure Rust implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    I32,
    U8,
}

impl DType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => std::mem::size_of::<f32>(),
            DType::I32 => std::mem::size_of::<i32>(),
            DType::U8 => std::mem::size_of::<u8>(),
        }
    }

    pub fn alignment(self) -> usize {
        self.size_in_bytes().max(1)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::I32 => write!(f, "i32"),
            DType::U8 => write!(f, "u8"),
        }
    }
}
