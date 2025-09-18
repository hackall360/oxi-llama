use ndarray::Array2;

#[cfg(feature = "tch")]
use std::convert::TryFrom;

#[cfg(feature = "tch")]
use tch::Tensor;

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

#[cfg(feature = "tch")]
#[derive(Default, Clone, Copy)]
pub struct TchBackend;

#[cfg(feature = "tch")]
impl TensorBackend for TchBackend {
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let a_tensor =
            Tensor::of_slice(a.as_slice().unwrap()).reshape(&[a.nrows() as i64, a.ncols() as i64]);
        let b_tensor =
            Tensor::of_slice(b.as_slice().unwrap()).reshape(&[b.nrows() as i64, b.ncols() as i64]);
        let out = a_tensor.matmul(&b_tensor);
        let shape = out.size();
        let vec: Vec<f32> = Vec::<f32>::try_from(out.reshape(&[-1])).unwrap();
        Array2::from_shape_vec((shape[0] as usize, shape[1] as usize), vec).unwrap()
    }
}

/// Supported backend variants.
#[derive(Clone, Copy, Debug)]
pub enum Backend {
    NdArray,
    #[cfg(feature = "tch")]
    Tch,
}

impl Backend {
    /// Construct a backend instance.
    pub fn build(self) -> Box<dyn TensorBackend> {
        match self {
            Backend::NdArray => Box::new(NdArrayBackend::default()),
            #[cfg(feature = "tch")]
            Backend::Tch => Box::new(TchBackend::default()),
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

    #[cfg(feature = "tch")]
    #[test]
    fn test_tch_backend_matmul() {
        let backend = TchBackend::default();
        let a = array![[1., 2.]];
        let b = array![[3.], [4.]];
        let out = backend.matmul(&a, &b);
        assert_eq!(out, array![[11.]]);
    }
}
