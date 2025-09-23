use packed_simd_2::{f32x16, f32x4, f32x8, Simd};

use crate::detection::capabilities;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdKind {
    #[cfg(feature = "avx")]
    Avx512,
    #[cfg(feature = "avx")]
    Avx2,
    #[cfg(feature = "neon")]
    Neon,
    Scalar,
}

impl SimdKind {
    #[inline]
    pub fn lanes(self) -> usize {
        match self {
            #[cfg(feature = "avx")]
            SimdKind::Avx512 => f32x16::lanes(),
            #[cfg(feature = "avx")]
            SimdKind::Avx2 => f32x8::lanes(),
            #[cfg(feature = "neon")]
            SimdKind::Neon => f32x4::lanes(),
            SimdKind::Scalar => 1,
        }
    }
}

#[inline]
pub fn select_simd() -> SimdKind {
    let caps = capabilities();
    #[cfg(feature = "avx")]
    {
        if caps.avx512 {
            return SimdKind::Avx512;
        }
        if caps.avx2 {
            return SimdKind::Avx2;
        }
    }
    #[cfg(feature = "neon")]
    {
        if caps.neon {
            return SimdKind::Neon;
        }
    }
    SimdKind::Scalar
}

#[inline]
pub fn fallback_dot(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

#[inline]
pub fn fallback_map_inplace(dst: &mut [f32], f: impl Fn(f32) -> f32 + Copy) {
    for v in dst {
        *v = f(*v);
    }
}

#[inline]
pub fn fallback_zip_inplace(dst: &mut [f32], rhs: &[f32], f: impl Fn(f32, f32) -> f32 + Copy) {
    for (d, r) in dst.iter_mut().zip(rhs.iter()) {
        *d = f(*d, *r);
    }
}

#[inline]
pub fn fallback_reduce(dst: &[f32], init: f32, f: impl Fn(f32, f32) -> f32) -> f32 {
    dst.iter().copied().fold(init, f)
}

#[inline]
pub fn apply_simd_unary<const LANES: usize, F, S>(dst: &mut [f32], simd_op: F, scalar_op: S)
where
    F: Fn(Simd<[f32; LANES]>) -> Simd<[f32; LANES]>,
    S: Fn(f32) -> f32,
{
    let mut i = 0;
    while i + LANES <= dst.len() {
        let chunk = Simd::<[f32; LANES]>::from_slice_unaligned(&dst[i..]);
        let res = simd_op(chunk);
        res.write_to_slice_unaligned(&mut dst[i..]);
        i += LANES;
    }
    while i < dst.len() {
        dst[i] = scalar_op(dst[i]);
        i += 1;
    }
}

#[inline]
pub fn apply_simd_binary<const LANES: usize, F, S>(
    dst: &mut [f32],
    rhs: &[f32],
    simd_op: F,
    scalar_op: S,
) where
    F: Fn(Simd<[f32; LANES]>, Simd<[f32; LANES]>) -> Simd<[f32; LANES]>,
    S: Fn(f32, f32) -> f32,
{
    let mut i = 0;
    while i + LANES <= dst.len() {
        let lhs = Simd::<[f32; LANES]>::from_slice_unaligned(&dst[i..]);
        let rhs_chunk = Simd::<[f32; LANES]>::from_slice_unaligned(&rhs[i..]);
        let res = simd_op(lhs, rhs_chunk);
        res.write_to_slice_unaligned(&mut dst[i..]);
        i += LANES;
    }
    while i < dst.len() {
        dst[i] = scalar_op(dst[i], rhs[i]);
        i += 1;
    }
}

#[inline]
pub fn dot_simd<const LANES: usize>(x: &[f32], y: &[f32]) -> f32 {
    let mut acc = Simd::<[f32; LANES]>::splat(0.0);
    let mut i = 0;
    while i + LANES <= x.len() {
        let a = Simd::<[f32; LANES]>::from_slice_unaligned(&x[i..]);
        let b = Simd::<[f32; LANES]>::from_slice_unaligned(&y[i..]);
        acc = acc + a * b;
        i += LANES;
    }
    let mut sum = acc.sum();
    while i < x.len() {
        sum += x[i] * y[i];
        i += 1;
    }
    sum
}
