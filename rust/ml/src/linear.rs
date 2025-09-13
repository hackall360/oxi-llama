use ndarray::{Array1, Array2, Axis};

/// Applies a linear transformation: `output = input * weight^T + bias`.
pub fn linear(input: &Array2<f32>, weight: &Array2<f32>, bias: Option<&Array1<f32>>) -> Array2<f32> {
    let mut output = input.dot(&weight.t());
    if let Some(b) = bias {
        output = output + &b.view().insert_axis(Axis(0));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear() {
        let input = array![[1., 2.]]; // (1,2)
        let weight = array![[1., 0.], [0., 1.]]; // (2,2)
        let bias = array![1., 1.];
        let out = linear(&input, &weight, Some(&bias));
        assert_eq!(out, array![[2., 3.]]);
    }
}
