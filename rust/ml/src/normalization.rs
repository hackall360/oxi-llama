use ndarray::{Array1, Array2, Axis};

/// Layer normalization following the transformer formulation.
pub fn layer_norm(
    input: &Array2<f32>,
    weight: Option<&Array1<f32>>,
    bias: Option<&Array1<f32>>,
    eps: f32,
) -> Array2<f32> {
    let mean = input.mean_axis(Axis(1)).unwrap();
    let var = input.var_axis(Axis(1), 0.0);
    let mut output = input.to_owned();
    for (mut row, (&m, &v)) in output
        .axis_iter_mut(Axis(0))
        .zip(mean.iter().zip(var.iter()))
    {
        let denom = (v + eps).sqrt();
        row.mapv_inplace(|x| (x - m) / denom);
    }
    if let Some(w) = weight {
        output = output * &w.view().insert_axis(Axis(0));
    }
    if let Some(b) = bias {
        output = output + &b.view().insert_axis(Axis(0));
    }
    output
}

/// Root mean square normalization.
pub fn rms_norm(input: &Array2<f32>, weight: Option<&Array1<f32>>, eps: f32) -> Array2<f32> {
    let mean_square = input.mapv(|x| x * x).mean_axis(Axis(1)).unwrap();
    let mut output = input.to_owned();
    for (mut row, &ms) in output.axis_iter_mut(Axis(0)).zip(mean_square.iter()) {
        let denom = (ms + eps).sqrt();
        row.mapv_inplace(|x| x / denom);
    }
    if let Some(w) = weight {
        output = output * &w.view().insert_axis(Axis(0));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_layer_norm_constant() {
        let input = array![[1., 1.], [1., 1.]];
        let out = layer_norm(&input, None, None, 1e-5);
        assert_eq!(out, array![[0., 0.], [0., 0.]]);
    }

    #[test]
    fn test_rms_norm_constant() {
        let input = array![[3., 4.]]; // RMS = 5
        let out = rms_norm(&input, None, 1e-6);
        let mean_sq = out.mapv(|x| x.powi(2)).mean().unwrap();
        assert!((mean_sq - 1.0).abs() < 1e-6);
    }
}
