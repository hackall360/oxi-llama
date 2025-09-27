use half::f16;

pub mod q4_0;

pub use q4_0::*;

#[inline(always)]
fn f32_to_f16(val: f32) -> f16 {
    f16::from_f32(val)
}

#[inline(always)]
fn f16_to_f32(val: f16) -> f32 {
    val.to_f32()
}
