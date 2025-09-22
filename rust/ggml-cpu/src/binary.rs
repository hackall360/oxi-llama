use crate::simd_common::{apply_simd_binary, fallback_zip_inplace, select_simd, SimdKind};

#[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
const AVX_LANES: usize = 16;
#[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
const AVX2_LANES: usize = 8;
#[cfg(all(feature = "neon", any(target_arch = "aarch64", target_arch = "arm")))]
const NEON_LANES: usize = 4;

/// Element-wise subtraction: `dst -= rhs`.
pub fn binary_sub_inplace(dst: &mut [f32], rhs: &[f32]) {
    assert_eq!(dst.len(), rhs.len());
    match select_simd() {
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx512 => {
            apply_simd_binary::<AVX_LANES, _, _>(dst, rhs, |a, b| a - b, |a, b| a - b);
        }
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx2 => {
            apply_simd_binary::<AVX2_LANES, _, _>(dst, rhs, |a, b| a - b, |a, b| a - b);
        }
        #[cfg(all(feature = "neon", any(target_arch = "aarch64", target_arch = "arm")))]
        SimdKind::Neon => {
            apply_simd_binary::<NEON_LANES, _, _>(dst, rhs, |a, b| a - b, |a, b| a - b);
        }
        _ => fallback_zip_inplace(dst, rhs, |a, b| a - b),
    }
}

/// Element-wise division: `dst /= rhs`.
pub fn binary_div_inplace(dst: &mut [f32], rhs: &[f32]) {
    assert_eq!(dst.len(), rhs.len());
    match select_simd() {
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx512 => {
            apply_simd_binary::<AVX_LANES, _, _>(dst, rhs, |a, b| a / b, |a, b| a / b);
        }
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx2 => {
            apply_simd_binary::<AVX2_LANES, _, _>(dst, rhs, |a, b| a / b, |a, b| a / b);
        }
        #[cfg(all(feature = "neon", any(target_arch = "aarch64", target_arch = "arm")))]
        SimdKind::Neon => {
            apply_simd_binary::<NEON_LANES, _, _>(dst, rhs, |a, b| a / b, |a, b| a / b);
        }
        _ => fallback_zip_inplace(dst, rhs, |a, b| a / b),
    }
}

/// Element-wise maximum: `dst = max(dst, rhs)`.
pub fn binary_max_inplace(dst: &mut [f32], rhs: &[f32]) {
    assert_eq!(dst.len(), rhs.len());
    match select_simd() {
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx512 => {
            apply_simd_binary::<AVX_LANES, _, _>(dst, rhs, |a, b| a.max(b), |a, b| a.max(b));
        }
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx2 => {
            apply_simd_binary::<AVX2_LANES, _, _>(dst, rhs, |a, b| a.max(b), |a, b| a.max(b));
        }
        #[cfg(all(feature = "neon", any(target_arch = "aarch64", target_arch = "arm")))]
        SimdKind::Neon => {
            apply_simd_binary::<NEON_LANES, _, _>(dst, rhs, |a, b| a.max(b), |a, b| a.max(b));
        }
        _ => fallback_zip_inplace(dst, rhs, f32::max),
    }
}

/// Element-wise minimum: `dst = min(dst, rhs)`.
pub fn binary_min_inplace(dst: &mut [f32], rhs: &[f32]) {
    assert_eq!(dst.len(), rhs.len());
    match select_simd() {
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx512 => {
            apply_simd_binary::<AVX_LANES, _, _>(dst, rhs, |a, b| a.min(b), |a, b| a.min(b));
        }
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx2 => {
            apply_simd_binary::<AVX2_LANES, _, _>(dst, rhs, |a, b| a.min(b), |a, b| a.min(b));
        }
        #[cfg(all(feature = "neon", any(target_arch = "aarch64", target_arch = "arm")))]
        SimdKind::Neon => {
            apply_simd_binary::<NEON_LANES, _, _>(dst, rhs, |a, b| a.min(b), |a, b| a.min(b));
        }
        _ => fallback_zip_inplace(dst, rhs, f32::min),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_sub() {
        let mut a = vec![3.0, 4.0, 5.0];
        let b = vec![1.0, 1.0, 1.0];
        binary_sub_inplace(&mut a, &b);
        assert_eq!(a, vec![2.0, 3.0, 4.0]);
    }
}
