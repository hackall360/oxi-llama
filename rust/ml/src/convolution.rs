use ndarray::{Array2, s};

/// A very small 2D convolution helper without padding or stride.
pub fn conv2d(input: &Array2<f32>, kernel: &Array2<f32>) -> Array2<f32> {
    let (h, w) = input.dim();
    let (kh, kw) = kernel.dim();
    let out_h = h - kh + 1;
    let out_w = w - kw + 1;
    let mut output = Array2::zeros((out_h, out_w));
    for i in 0..out_h {
        for j in 0..out_w {
            let window = input.slice(s![i..i + kh, j..j + kw]);
            output[(i, j)] = (&window * kernel).sum();
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_conv2d() {
        let input = array![
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]
        ];
        let kernel = array![[1., 0.], [0., -1.]];
        let out = conv2d(&input, &kernel);
        assert_eq!(out, array![[-4., -4.], [-4., -4.]]);
    }
}
