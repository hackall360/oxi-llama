use ndarray::{Array1, Array2};

/// Simple embedding lookup.
pub fn embedding(weight: &Array2<f32>, indices: &Array1<usize>) -> Array2<f32> {
    let mut out = Array2::zeros((indices.len(), weight.shape()[1]));
    for (i, &idx) in indices.iter().enumerate() {
        out.row_mut(i).assign(&weight.row(idx));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_embedding_lookup() {
        let weight = array![[1., 2.], [3., 4.], [5., 6.]];
        let idx = array![2usize, 0usize];
        let out = embedding(&weight, &idx);
        assert_eq!(out, array![[5., 6.], [1., 2.]]);
    }
}
