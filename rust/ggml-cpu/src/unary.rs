use crate::simd_common::{apply_simd_unary, fallback_map_inplace, select_simd, SimdKind};
use packed_simd_2::Simd;

#[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
const AVX_LANES: usize = 16;
#[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
const AVX2_LANES: usize = 8;
#[cfg(all(feature = "neon", any(target_arch = "aarch64", target_arch = "arm")))]
const NEON_LANES: usize = 4;

fn map_simd<const LANES: usize>(
    value: Simd<[f32; LANES]>,
    op: fn(f32) -> f32,
) -> Simd<[f32; LANES]> {
    let mut arr = value.to_array();
    for lane in &mut arr {
        *lane = op(*lane);
    }
    Simd::from_array(arr)
}

fn dispatch_unary(dst: &mut [f32], simd: SimdKind, op: fn(f32) -> f32) {
    match simd {
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx512 => {
            apply_simd_unary::<AVX_LANES, _, _>(dst, |v| map_simd(v, op), op);
        }
        #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
        SimdKind::Avx2 => {
            apply_simd_unary::<AVX2_LANES, _, _>(dst, |v| map_simd(v, op), op);
        }
        #[cfg(all(feature = "neon", any(target_arch = "aarch64", target_arch = "arm")))]
        SimdKind::Neon => {
            apply_simd_unary::<NEON_LANES, _, _>(dst, |v| map_simd(v, op), op);
        }
        _ => fallback_map_inplace(dst, op),
    }
}

/// Rectified Linear Unit activation.
pub fn unary_relu_inplace(dst: &mut [f32]) {
    let simd = select_simd();
    dispatch_unary(dst, simd, |x| x.max(0.0));
}

/// Sigmoid Linear Unit activation (a.k.a. SiLU).
pub fn unary_silu_inplace(dst: &mut [f32]) {
    let simd = select_simd();
    dispatch_unary(dst, simd, |x| x / (1.0 + (-x).exp()));
}

/// Gaussian Error Linear Unit activation using the tanh approximation.
pub fn unary_gelu_inplace(dst: &mut [f32]) {
    let simd = select_simd();
    dispatch_unary(dst, simd, |x| {
        0.5 * x * (1.0 + (0.79788456 * (x + 0.044715 * x.powi(3))).tanh())
    });
}

/// Hyperbolic tangent activation.
pub fn unary_tanh_inplace(dst: &mut [f32]) {
    let simd = select_simd();
    dispatch_unary(dst, simd, |x| x.tanh());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unary_relu() {
        let mut data = vec![-1.0, 0.5, 2.0];
        unary_relu_inplace(&mut data);
        assert_eq!(data, vec![0.0, 0.5, 2.0]);
    }
}
