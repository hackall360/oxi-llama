use crate::simd_common::{apply_simd_binary, apply_simd_unary, dot_simd};

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dot_f32_avx2_impl(x: &[f32], y: &[f32]) -> f32 {
    dot_simd::<8>(x, y)
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_avx512_impl(x: &[f32], y: &[f32]) -> f32 {
    dot_simd::<16>(x, y)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn add_inplace_f32_avx2_impl(dst: &mut [f32], rhs: &[f32]) {
    apply_simd_binary::<8, _, _>(dst, rhs, |a, b| a + b, |a, b| a + b);
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn add_inplace_f32_avx512_impl(dst: &mut [f32], rhs: &[f32]) {
    apply_simd_binary::<16, _, _>(dst, rhs, |a, b| a + b, |a, b| a + b);
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mul_inplace_f32_avx2_impl(dst: &mut [f32], rhs: &[f32]) {
    apply_simd_binary::<8, _, _>(dst, rhs, |a, b| a * b, |a, b| a * b);
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn mul_inplace_f32_avx512_impl(dst: &mut [f32], rhs: &[f32]) {
    apply_simd_binary::<16, _, _>(dst, rhs, |a, b| a * b, |a, b| a * b);
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn scale_inplace_f32_avx2_impl(dst: &mut [f32], value: f32) {
    apply_simd_unary::<8, _, _>(dst, |a| a * value, |a| a * value);
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn scale_inplace_f32_avx512_impl(dst: &mut [f32], value: f32) {
    apply_simd_unary::<16, _, _>(dst, |a| a * value, |a| a * value);
}

#[inline]
pub fn dot_f32_avx2(x: &[f32], y: &[f32]) -> f32 {
    unsafe { dot_f32_avx2_impl(x, y) }
}

#[inline]
pub fn dot_f32_avx512(x: &[f32], y: &[f32]) -> f32 {
    unsafe { dot_f32_avx512_impl(x, y) }
}

#[inline]
pub fn add_inplace_f32_avx2(dst: &mut [f32], rhs: &[f32]) {
    unsafe { add_inplace_f32_avx2_impl(dst, rhs) }
}

#[inline]
pub fn add_inplace_f32_avx512(dst: &mut [f32], rhs: &[f32]) {
    unsafe { add_inplace_f32_avx512_impl(dst, rhs) }
}

#[inline]
pub fn mul_inplace_f32_avx2(dst: &mut [f32], rhs: &[f32]) {
    unsafe { mul_inplace_f32_avx2_impl(dst, rhs) }
}

#[inline]
pub fn mul_inplace_f32_avx512(dst: &mut [f32], rhs: &[f32]) {
    unsafe { mul_inplace_f32_avx512_impl(dst, rhs) }
}

#[inline]
pub fn scale_inplace_f32_avx2(dst: &mut [f32], value: f32) {
    unsafe { scale_inplace_f32_avx2_impl(dst, value) }
}

#[inline]
pub fn scale_inplace_f32_avx512(dst: &mut [f32], value: f32) {
    unsafe { scale_inplace_f32_avx512_impl(dst, value) }
}
