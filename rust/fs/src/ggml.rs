/// Basic tensor representation used by the GGUF writer.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<u64>,
    /// GGML data type.  For the purposes of the tests we only need f32 which
    /// corresponds to `0` in the GGUF specification.
    pub kind: u32,
    /// Raw tensor bytes in little endian order.
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn new<N: Into<String>>(name: N, shape: Vec<u64>, data: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            shape,
            kind: 0,
            data,
        }
    }

    /// Returns the number of bytes used by the tensor data.
    pub fn num_bytes(&self) -> u64 {
        self.data.len() as u64
    }
}
