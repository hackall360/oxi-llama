use ndarray::{Array1, Array2};

/// Applies rotary positional embeddings (RoPE) to the given tensor.
///
/// `x` has shape `(seq_len, dim)` where `dim` is even. `positions` is a
/// vector of length `seq_len` describing the token positions.
pub fn apply_rope(x: &Array2<f32>, positions: &Array1<f32>) -> Array2<f32> {
    let (seq_len, dim) = x.dim();
    assert_eq!(positions.len(), seq_len);
    assert!(dim % 2 == 0, "dimension must be even");
    let half = dim / 2;
    let mut out = x.clone();
    for (t, &pos) in positions.iter().enumerate() {
        for i in 0..half {
            let freq = (2 * i) as f32 / dim as f32;
            let theta = pos / 10000f32.powf(freq);
            let cos = theta.cos();
            let sin = theta.sin();
            let x0 = x[(t, i)];
            let x1 = x[(t, i + half)];
            out[(t, i)] = x0 * cos - x1 * sin;
            out[(t, i + half)] = x0 * sin + x1 * cos;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_rope_preserves_norm() {
        let x = array![[1., 0., 0., 1.]]; // seq_len=1, dim=4
        let pos = array![1.0];
        let out = apply_rope(&x, &pos);
        let norm_before: f32 = x.iter().map(|v| v * v).sum();
        let norm_after: f32 = out.iter().map(|v| v * v).sum();
        assert!((norm_before - norm_after).abs() < 1e-6);
    }
}
