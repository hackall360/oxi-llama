use crate::simd_common::{
    fallback_dot, fallback_map_inplace, fallback_reduce, fallback_zip_inplace, select_simd,
    SimdKind,
};

#[cfg(all(feature = "neon", any(target_arch = "aarch64", target_arch = "arm")))]
use crate::arch::arm;
#[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::arch::x86;

/// Compute the dot product of two `f32` slices using the best available SIMD
/// implementation for the current CPU.
pub fn vec_dot_f32(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    match select_simd() {
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx512 => x86::dot_f32_avx512(x, y),
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx2 => x86::dot_f32_avx2(x, y),
        #[cfg(all(feature = "neon", any(target_arch = "aarch64", target_arch = "arm")))]
        SimdKind::Neon => arm::dot_f32_neon(x, y),
        _ => fallback_dot(x, y),
    }
}

/// In-place addition: `dst += rhs`.
pub fn vec_add_inplace(dst: &mut [f32], rhs: &[f32]) {
    assert_eq!(dst.len(), rhs.len());
    match select_simd() {
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx512 => x86::add_inplace_f32_avx512(dst, rhs),
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx2 => x86::add_inplace_f32_avx2(dst, rhs),
        #[cfg(all(feature = "neon", any(target_arch = "aarch64", target_arch = "arm")))]
        SimdKind::Neon => arm::add_inplace_f32_neon(dst, rhs),
        _ => fallback_zip_inplace(dst, rhs, |a, b| a + b),
    }
}

/// In-place multiplication: `dst *= rhs`.
pub fn vec_mul_inplace(dst: &mut [f32], rhs: &[f32]) {
    assert_eq!(dst.len(), rhs.len());
    match select_simd() {
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx512 => x86::mul_inplace_f32_avx512(dst, rhs),
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx2 => x86::mul_inplace_f32_avx2(dst, rhs),
        #[cfg(all(feature = "neon", any(target_arch = "aarch64", target_arch = "arm")))]
        SimdKind::Neon => arm::mul_inplace_f32_neon(dst, rhs),
        _ => fallback_zip_inplace(dst, rhs, |a, b| a * b),
    }
}

/// Scale `dst` by a scalar value.
pub fn vec_scale_inplace(dst: &mut [f32], value: f32) {
    match select_simd() {
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx512 => x86::scale_inplace_f32_avx512(dst, value),
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx2 => x86::scale_inplace_f32_avx2(dst, value),
        #[cfg(all(feature = "neon", any(target_arch = "aarch64", target_arch = "arm")))]
        SimdKind::Neon => arm::scale_inplace_f32_neon(dst, value),
        _ => fallback_map_inplace(dst, |x| x * value),
    }
}

/// Sum of squares of `dst` (used as building block for norms).
pub fn vec_sum_squares(dst: &[f32]) -> f32 {
    match select_simd() {
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx512 => {
            let mut tmp = dst.to_vec();
            vec_mul_inplace(&mut tmp, dst);
            tmp.into_iter().sum()
        }
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx2 => {
            let mut tmp = dst.to_vec();
            vec_mul_inplace(&mut tmp, dst);
            tmp.into_iter().sum()
        }
        #[cfg(all(feature = "neon", any(target_arch = "aarch64", target_arch = "arm")))]
        SimdKind::Neon => {
            let mut tmp = dst.to_vec();
            vec_mul_inplace(&mut tmp, dst);
            tmp.into_iter().sum()
        }
        _ => dst.iter().map(|v| v * v).sum(),
    }
}

/// Compute the L2 norm of `dst`.
pub fn vec_norm_f32(dst: &[f32]) -> f32 {
    vec_sum_squares(dst).sqrt()
}

/// Compute a sum reduction over `dst`.
pub fn vec_sum(dst: &[f32]) -> f32 {
    fallback_reduce(dst, 0.0, |acc, v| acc + v)
}

/// Compute the maximum absolute value.
pub fn vec_max_abs(dst: &[f32]) -> f32 {
    dst.iter().map(|v| v.abs()).fold(0.0, f32::max)
}

/// Normalize the values in `dst` so that they sum to one.
pub fn vec_softmax_inplace(dst: &mut [f32]) {
    if dst.is_empty() {
        return;
    }
    let max = dst.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    fallback_map_inplace(dst, |v| (v - max).exp());
    let sum = vec_sum(dst);
    if sum > 0.0 {
        vec_scale_inplace(dst, 1.0 / sum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_vec_dot() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        assert_relative_eq!(vec_dot_f32(&a, &b), 70.0, epsilon = 1e-5);
    }

    #[test]
    fn test_vec_add_inplace() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        vec_add_inplace(&mut a, &b);
        assert_eq!(a, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_vec_softmax() {
        let mut v = vec![0.0, 0.0, 0.0];
        vec_softmax_inplace(&mut v);
        assert_relative_eq!(v[0], v[1], epsilon = 1e-6);
        assert_relative_eq!(v.iter().sum::<f32>(), 1.0, epsilon = 1e-5);
    }
}
