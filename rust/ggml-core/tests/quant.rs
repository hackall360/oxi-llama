use approx::assert_relative_eq;
use ggml_core::quant::{
    dequantize_row_q4_0, dequantize_row_q8_0, quantize_row_q4_0_ref, quantize_row_q8_0_ref,
    vec_dot_q4_0_q8_0, BlockQ4_0, BlockQ8_0, QK4_0, QK8_0,
};

extern "C" {
    fn quant_ref_quantize_row_q4_0(x: *const f32, y: *mut BlockQ4_0, k: i64);
    fn quant_ref_quantize_row_q8_0(x: *const f32, y: *mut BlockQ8_0, k: i64);
    fn quant_ref_dequantize_row_q4_0(x: *const BlockQ4_0, y: *mut f32, k: i64);
    fn quant_ref_dequantize_row_q8_0(x: *const BlockQ8_0, y: *mut f32, k: i64);
    fn quant_ref_vec_dot_q4_0_q8_0(x: *const BlockQ4_0, y: *const BlockQ8_0, n: i64) -> f32;
}

fn sample_data(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * 0.123).sin() * 1.5 + ((i as f32) * 0.321).cos() * 0.75)
        .collect()
}

fn blocks_from_c<T: Default + Clone>(len: usize) -> Vec<T> {
    vec![T::default(); len]
}

#[test]
fn quantize_q4_0_matches_c() {
    const N: usize = 4 * QK4_0;
    let input = sample_data(N);

    let mut rust_blocks = blocks_from_c::<BlockQ4_0>(N / QK4_0);
    quantize_row_q4_0_ref(&input, &mut rust_blocks);

    let mut c_blocks = blocks_from_c::<BlockQ4_0>(N / QK4_0);
    unsafe {
        quant_ref_quantize_row_q4_0(input.as_ptr(), c_blocks.as_mut_ptr(), N as i64);
    }

    for (r, c) in rust_blocks.iter().zip(c_blocks.iter()) {
        assert_eq!(r.d.to_bits(), c.d.to_bits());
        assert_eq!(r.qs, c.qs);
    }
}

#[test]
fn dequantize_q4_0_matches_c() {
    const N: usize = 4 * QK4_0;
    let input = sample_data(N);

    let mut blocks = blocks_from_c::<BlockQ4_0>(N / QK4_0);
    quantize_row_q4_0_ref(&input, &mut blocks);

    let mut rust_out = vec![0.0f32; N];
    dequantize_row_q4_0(&blocks, &mut rust_out);

    let mut c_out = vec![0.0f32; N];
    unsafe {
        quant_ref_dequantize_row_q4_0(blocks.as_ptr(), c_out.as_mut_ptr(), N as i64);
    }

    for (r, c) in rust_out.iter().zip(c_out.iter()) {
        assert_relative_eq!(r, c, epsilon = 1e-5f32);
    }
}

#[test]
fn quantize_q8_0_matches_c() {
    const N: usize = 4 * QK8_0;
    let input = sample_data(N);

    let mut rust_blocks = blocks_from_c::<BlockQ8_0>(N / QK8_0);
    quantize_row_q8_0_ref(&input, &mut rust_blocks);

    let mut c_blocks = blocks_from_c::<BlockQ8_0>(N / QK8_0);
    unsafe {
        quant_ref_quantize_row_q8_0(input.as_ptr(), c_blocks.as_mut_ptr(), N as i64);
    }

    for (r, c) in rust_blocks.iter().zip(c_blocks.iter()) {
        assert_eq!(r.d.to_bits(), c.d.to_bits());
        assert_eq!(r.qs, c.qs);
    }
}

#[test]
fn dequantize_q8_0_matches_c() {
    const N: usize = 4 * QK8_0;
    let input = sample_data(N);

    let mut blocks = blocks_from_c::<BlockQ8_0>(N / QK8_0);
    quantize_row_q8_0_ref(&input, &mut blocks);

    let mut rust_out = vec![0.0f32; N];
    dequantize_row_q8_0(&blocks, &mut rust_out);

    let mut c_out = vec![0.0f32; N];
    unsafe {
        quant_ref_dequantize_row_q8_0(blocks.as_ptr(), c_out.as_mut_ptr(), N as i64);
    }

    for (r, c) in rust_out.iter().zip(c_out.iter()) {
        assert_relative_eq!(r, c, epsilon = 1e-5f32);
    }
}

#[test]
fn vec_dot_q4_0_q8_0_matches_c() {
    const N: usize = 4 * QK4_0;
    let x_data = sample_data(N);
    let y_data = sample_data(N)
        .into_iter()
        .map(|v| v * 0.5)
        .collect::<Vec<_>>();

    let mut x_blocks = blocks_from_c::<BlockQ4_0>(N / QK4_0);
    let mut y_blocks = blocks_from_c::<BlockQ8_0>(N / QK8_0);
    quantize_row_q4_0_ref(&x_data, &mut x_blocks);
    quantize_row_q8_0_ref(&y_data, &mut y_blocks);

    let rust = vec_dot_q4_0_q8_0(&x_blocks, &y_blocks);
    let c_val =
        unsafe { quant_ref_vec_dot_q4_0_q8_0(x_blocks.as_ptr(), y_blocks.as_ptr(), N as i64) };

    assert_relative_eq!(rust, c_val, epsilon = 1e-3f32);
}
