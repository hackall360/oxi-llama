use ndarray::{Array2, Axis};

/// Simple scaled dot-product attention implementation using `ndarray`.
///
/// `query`, `key` and `value` are all matrices where the first axis is the
/// sequence dimension and the second axis is the feature dimension.
pub fn scaled_dot_product_attention(
    query: &Array2<f32>,
    key: &Array2<f32>,
    value: &Array2<f32>,
    scale: f32,
) -> Array2<f32> {
    let mut scores = query.dot(&key.t());
    scores *= scale;
    // softmax along last axis
    scores.mapv_inplace(|x| x.exp());
    let sums = scores.sum_axis(Axis(1)).insert_axis(Axis(1));
    let probs = scores / &sums;
    probs.dot(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_attention_shapes() {
        let q = array![[1., 0.]]; // (1,2)
        let k = array![[1., 0.], [0., 1.]]; // (2,2)
        let v = array![[1., 2.], [3., 4.]]; // (2,2)
        let out = scaled_dot_product_attention(&q, &k, &v, 1.0);
        assert_eq!(out.shape(), &[1, 2]);
    }
}
