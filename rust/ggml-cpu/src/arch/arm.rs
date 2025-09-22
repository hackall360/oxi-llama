use crate::simd_common::{apply_simd_binary, apply_simd_unary, dot_simd};

#[inline]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon_impl(x: &[f32], y: &[f32]) -> f32 {
    dot_simd::<4>(x, y)
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn add_inplace_f32_neon_impl(dst: &mut [f32], rhs: &[f32]) {
    apply_simd_binary::<4, _, _>(dst, rhs, |a, b| a + b, |a, b| a + b);
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn mul_inplace_f32_neon_impl(dst: &mut [f32], rhs: &[f32]) {
    apply_simd_binary::<4, _, _>(dst, rhs, |a, b| a * b, |a, b| a * b);
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn scale_inplace_f32_neon_impl(dst: &mut [f32], value: f32) {
    apply_simd_unary::<4, _, _>(dst, |a| a * value, |a| a * value);
}

#[inline]
pub fn dot_f32_neon(x: &[f32], y: &[f32]) -> f32 {
    unsafe { dot_f32_neon_impl(x, y) }
}

#[inline]
pub fn add_inplace_f32_neon(dst: &mut [f32], rhs: &[f32]) {
    unsafe { add_inplace_f32_neon_impl(dst, rhs) }
}

#[inline]
pub fn mul_inplace_f32_neon(dst: &mut [f32], rhs: &[f32]) {
    unsafe { mul_inplace_f32_neon_impl(dst, rhs) }
}

#[inline]
pub fn scale_inplace_f32_neon(dst: &mut [f32], value: f32) {
    unsafe { scale_inplace_f32_neon_impl(dst, value) }
}
