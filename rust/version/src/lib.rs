#![deny(missing_docs)]

//! Version and build metadata helpers for Oxi.

/// Version string for the currently built crate.
#[allow(non_upper_case_globals)]
pub const Version: &str = env!("OXI_VERSION");

/// Git commit hash embedded into the build.
pub const GIT_COMMIT: &str = env!("OXI_GIT_COMMIT");

const fn parse_bool(input: &str) -> bool {
    eq_ignore_case(input, "1")
        || eq_ignore_case(input, "true")
        || eq_ignore_case(input, "t")
        || eq_ignore_case(input, "yes")
        || eq_ignore_case(input, "y")
}

const fn eq_ignore_case(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let mut i = 0;
    while i < a_bytes.len() {
        if to_lower(a_bytes[i]) != to_lower(b_bytes[i]) {
            return false;
        }
        i += 1;
    }
    true
}

const fn to_lower(b: u8) -> u8 {
    if b >= b'A' && b <= b'Z' {
        b + 32
    } else {
        b
    }
}

/// Indicates if the working tree was dirty when the build was produced.
pub const GIT_DIRTY: bool = parse_bool(env!("OXI_GIT_DIRTY"));

/// Unix timestamp (seconds) when the build was produced.
pub const BUILD_TIMESTAMP: &str = env!("OXI_BUILD_TIMESTAMP");

/// Target triple used to compile the build.
pub const BUILD_TARGET: &str = env!("OXI_BUILD_TARGET");

/// Cargo profile used for compilation.
pub const BUILD_PROFILE: &str = env!("OXI_BUILD_PROFILE");

/// Rust compiler version used to produce the build.
pub const RUSTC_VERSION: &str = env!("OXI_RUSTC_VERSION");

/// Aggregated build metadata describing the binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BuildMetadata {
    /// Semantic version string.
    pub version: &'static str,
    /// Git commit hash.
    pub git_commit: &'static str,
    /// Indicates if the source tree was dirty.
    pub git_dirty: bool,
    /// Unix timestamp for the build.
    pub build_timestamp: &'static str,
    /// Compilation target triple.
    pub build_target: &'static str,
    /// Cargo profile used for compilation.
    pub build_profile: &'static str,
    /// Rust compiler version.
    pub rustc_version: &'static str,
}

/// Build metadata for the current crate.
pub const BUILD_METADATA: BuildMetadata = BuildMetadata {
    version: Version,
    git_commit: GIT_COMMIT,
    git_dirty: GIT_DIRTY,
    build_timestamp: BUILD_TIMESTAMP,
    build_target: BUILD_TARGET,
    build_profile: BUILD_PROFILE,
    rustc_version: RUSTC_VERSION,
};

/// Returns build metadata for the current crate.
#[must_use]
pub const fn metadata() -> BuildMetadata {
    BUILD_METADATA
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_matches_constants() {
        let meta = metadata();
        assert_eq!(meta.version, Version);
        assert_eq!(meta.git_commit, GIT_COMMIT);
        assert_eq!(meta.git_dirty, GIT_DIRTY);
        assert_eq!(meta.build_timestamp, BUILD_TIMESTAMP);
        assert_eq!(meta.build_target, BUILD_TARGET);
        assert_eq!(meta.build_profile, BUILD_PROFILE);
        assert_eq!(meta.rustc_version, RUSTC_VERSION);
    }
}
