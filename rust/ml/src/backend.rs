use ndarray::Array2;

/// Trait representing basic tensor operations needed by the layers.
pub trait TensorBackend {
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;
}

/// Backend implemented with the `ndarray` crate.
#[derive(Default, Clone, Copy)]
pub struct NdArrayBackend;

impl TensorBackend for NdArrayBackend {
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        a.dot(b)
    }
}

/// Supported backend variants.
#[derive(Clone, Copy, Debug)]
pub enum Backend {
    NdArray,
}

impl Backend {
    /// Construct a backend instance.
    pub fn build(self) -> Box<dyn TensorBackend> {
        match self {
            Backend::NdArray => Box::new(NdArrayBackend::default()),
        }
    }

    /// Default backend used when none is specified.
    pub fn default() -> Self {
        Backend::NdArray
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_backend_matmul() {
        let backend = NdArrayBackend::default();
        let a = array![[1., 2.]];
        let b = array![[3.], [4.]];
        let out = backend.matmul(&a, &b);
        assert_eq!(out, array![[11.]]);
    }
}
