use crate::quant::{f16_to_f32, f32_to_f16};
use half::f16;

pub const QK4_0: usize = 32;
pub const QK8_0: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ4_0 {
    pub d: f16,
    pub qs: [u8; QK4_0 / 2],
}

impl Default for BlockQ4_0 {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            qs: [0; QK4_0 / 2],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ8_0 {
    pub d: f16,
    pub qs: [i8; QK8_0],
}

impl Default for BlockQ8_0 {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            qs: [0; QK8_0],
        }
    }
}

pub fn quantize_row_q4_0_ref(x: &[f32], y: &mut [BlockQ4_0]) {
    assert_eq!(
        x.len(),
        y.len() * QK4_0,
        "input length must equal blocks * QK4_0"
    );

    for (src, dst) in x.chunks_exact(QK4_0).zip(y.iter_mut()) {
        let mut amax = 0.0f32;
        let mut max = 0.0f32;

        for &v in src.iter() {
            let absv = v.abs();
            if absv > amax {
                amax = absv;
                max = v;
            }
        }

        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        dst.d = f32_to_f16(d);

        for (j, dst_q) in dst.qs.iter_mut().enumerate() {
            let x0 = src[j] * id;
            let x1 = src[j + QK4_0 / 2] * id;

            let xi0 = ((x0 + 8.5).trunc() as i8).min(15) as u8;
            let xi1 = ((x1 + 8.5).trunc() as i8).min(15) as u8;

            *dst_q = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        }
    }
}

pub fn dequantize_row_q4_0(x: &[BlockQ4_0], y: &mut [f32]) {
    assert_eq!(y.len(), x.len() * QK4_0);

    for (block, dst) in x.iter().zip(y.chunks_exact_mut(QK4_0)) {
        let d = f16_to_f32(block.d);
        for j in 0..QK4_0 / 2 {
            let v0 = ((block.qs[j] & 0x0f) as i32) - 8;
            let v1 = ((block.qs[j] >> 4) as i32) - 8;

            dst[j] = (v0 as f32) * d;
            dst[j + QK4_0 / 2] = (v1 as f32) * d;
        }
    }
}

pub fn quantize_row_q8_0_ref(x: &[f32], y: &mut [BlockQ8_0]) {
    assert_eq!(x.len(), y.len() * QK8_0);

    for (src, dst) in x.chunks_exact(QK8_0).zip(y.iter_mut()) {
        let mut amax = 0.0f32;
        for &v in src.iter() {
            amax = amax.max(v.abs());
        }

        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        dst.d = f32_to_f16(d);

        for (q, &val) in dst.qs.iter_mut().zip(src.iter()) {
            let scaled = (val * id).round();
            let clamped = scaled.clamp(-128.0, 127.0) as i32;
            *q = clamped as i8;
        }
    }
}

pub fn dequantize_row_q8_0(x: &[BlockQ8_0], y: &mut [f32]) {
    assert_eq!(y.len(), x.len() * QK8_0);

    for (block, dst) in x.iter().zip(y.chunks_exact_mut(QK8_0)) {
        let d = f16_to_f32(block.d);
        for (val, out) in block.qs.iter().zip(dst.iter_mut()) {
            *out = (*val as f32) * d;
        }
    }
}

pub fn vec_dot_q4_0_q8_0(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    assert_eq!(x.len(), y.len());

    let mut sumf = 0.0f32;
    for (xb, yb) in x.iter().zip(y.iter()) {
        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;

        for j in 0..QK4_0 / 2 {
            let v0 = (xb.qs[j] & 0x0F) as i32 - 8;
            let v1 = (xb.qs[j] >> 4) as i32 - 8;

            sumi0 += v0 * yb.qs[j] as i32;
            sumi1 += v1 * yb.qs[j + QK8_0 / 2] as i32;
        }

        let d = f16_to_f32(xb.d) * f16_to_f32(yb.d);
        sumf += (sumi0 + sumi1) as f32 * d;
    }

    sumf
}
