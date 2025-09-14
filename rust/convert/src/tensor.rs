use std::io::{self, Write};

/// Tensor data type used during conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKind {
    F32,
    F16,
    BF16,
    MXFP4,
}

/// Function used to repack tensor data prior to writing.
pub type Repacker = fn(&str, &[f32], &[u64]) -> io::Result<Vec<f32>>;

/// Trait representing tensor operations used by the converter.
pub trait Tensor: Send {
    fn name(&self) -> &str;
    fn shape(&self) -> &[u64];
    fn kind(&self) -> TensorKind;
    fn set_repacker(&mut self, repacker: Repacker);
    fn write_to(&self, w: &mut dyn Write) -> io::Result<()>;
    fn clone_box(&self) -> Box<dyn Tensor>;
}

impl Clone for Box<dyn Tensor> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Basic in-memory tensor implementation used for tests and stubs.
#[derive(Clone)]
pub struct BaseTensor {
    name: String,
    shape: Vec<u64>,
    data: Vec<f32>,
    repacker: Option<Repacker>,
}

impl BaseTensor {
    pub fn new<N: Into<String>>(name: N, shape: Vec<u64>, data: Vec<f32>) -> Self {
        Self { name: name.into(), shape, data, repacker: None }
    }
}

impl Tensor for BaseTensor {
    fn name(&self) -> &str { &self.name }
    fn shape(&self) -> &[u64] { &self.shape }
    fn kind(&self) -> TensorKind {
        if self.name.ends_with(".ffn_gate_inp.weight")
            || self.name.ends_with(".bias")
            || self.name == "token_types.weight"
            || self.name == "v.positional_embedding_vlm"
            || self.name == "v.tile_position_embd.weight"
            || self.name == "v.pre_tile_position_embd.weight"
            || self.name == "v.post_tile_position_embd.weight" {
            return TensorKind::F32;
        }
        match self.shape.len() {
            0 => panic!("invalid tensor shape"),
            1 => TensorKind::F32,
            _ => TensorKind::F16,
        }
    }
    fn set_repacker(&mut self, repacker: Repacker) {
        self.repacker = Some(repacker);
    }
    fn write_to(&self, w: &mut dyn Write) -> io::Result<()> {
        let mut data = self.data.clone();
        if let Some(rp) = self.repacker {
            data = rp(&self.name, &data, &self.shape)?;
        }
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for f in data {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        w.write_all(&bytes)
    }
    fn clone_box(&self) -> Box<dyn Tensor> { Box::new(self.clone()) }
}
