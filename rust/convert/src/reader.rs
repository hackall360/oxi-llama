use std::{fs, path::PathBuf};

use anyhow::{bail, Result};

use crate::tensor::{BaseTensor, Tensor};

/// Trait representing an abstract tensor reader.
pub trait ModelReader {
    fn read_tensors(&self) -> Result<Vec<Box<dyn Tensor>>>;
}

/// Trait representing an abstract tensor writer.
pub trait ModelWriter {
    fn write_tensor(&mut self, tensor: &dyn Tensor) -> Result<()>;
}

/// Simple filesystem based reader used for tests.
#[derive(Debug, Clone)]
pub struct FsReader {
    root: PathBuf,
}

impl FsReader {
    pub fn new<P: Into<PathBuf>>(root: P) -> Self {
        Self { root: root.into() }
    }
}

impl ModelReader for FsReader {
    fn read_tensors(&self) -> Result<Vec<Box<dyn Tensor>>> {
        let mut out: Vec<Box<dyn Tensor>> = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("tensor") {
                // fake tensor file consisting of f32 values
                let data = fs::read(&path)?;
                let mut f32s = Vec::new();
                for chunk in data.chunks(4) {
                    f32s.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
                let t = BaseTensor::new(
                    path.file_stem().unwrap().to_string_lossy(),
                    vec![f32s.len() as u64],
                    f32s,
                );
                out.push(Box::new(t));
            }
        }
        if out.is_empty() {
            bail!("unknown tensor format");
        }
        Ok(out)
    }
}

/// In-memory writer collecting tensors.
#[derive(Default)]
pub struct VecWriter {
    pub tensors: Vec<Box<dyn Tensor>>,
}

impl ModelWriter for VecWriter {
    fn write_tensor(&mut self, tensor: &dyn Tensor) -> Result<()> {
        self.tensors.push(tensor.clone_box());
        Ok(())
    }
}
