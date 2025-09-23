use once_cell::sync::Lazy;

/// Runtime CPU feature information collected once per process.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuCapabilities {
    pub avx: bool,
    pub avx2: bool,
    pub avx512: bool,
    pub fma: bool,
    pub amx_int8: bool,
    pub neon: bool,
}

static CAPS: Lazy<CpuCapabilities> = Lazy::new(|| {
    #[allow(unused_mut)]
    let mut caps = CpuCapabilities::default();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use std::arch::is_x86_feature_detected;
        caps.avx = is_x86_feature_detected!("avx");
        caps.avx2 = is_x86_feature_detected!("avx2");
        caps.avx512 = is_x86_feature_detected!("avx512f");
        caps.fma = is_x86_feature_detected!("fma");
        caps.amx_int8 =
            is_x86_feature_detected!("amx_int8") && is_x86_feature_detected!("amx_tile");
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected;
        caps.neon = is_aarch64_feature_detected!("neon");
    }

    #[cfg(target_arch = "arm")]
    {
        // 32-bit ARM exposes NEON through is_arm_feature_detected.
        use std::arch::is_arm_feature_detected;
        caps.neon = is_arm_feature_detected!("neon");
    }

    caps
});

/// Returns the lazily computed CPU capabilities.
#[inline]
pub fn capabilities() -> &'static CpuCapabilities {
    &CAPS
}
