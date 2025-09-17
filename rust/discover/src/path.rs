use std::path::{Path, PathBuf};

/// Determine the base path used to locate bundled dynamic libraries.
///
/// The behaviour mirrors the Go implementation in `discover/path.go` and
/// tries several locations depending on the target operating system.
///
/// The search order is:
/// 1. Platform specific library directory next to the executable.
/// 2. `build/lib/ollama` relative to the executable directory.
/// 3. `build/lib/ollama` relative to the current working directory.
/// 4. Finally fall back to the executable directory itself.
pub fn lib_ollama_path() -> PathBuf {
    let exe_path = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::canonicalize(&p).ok().or(Some(p)))
        .unwrap_or_else(|| PathBuf::from("."));

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    determine_lib_ollama_path(&exe_path, &cwd, |p| p.is_dir())
}

/// Construct default dependency paths for a GPU entry based on the discovered
/// Ollama library directory.
///
/// If a variant is supplied (for example CUDA driver versions), the variant
/// directory is returned before the base directory so that callers can probe
/// specialised payloads first.
pub fn default_dependency_paths(library: &str, variant: &str) -> Vec<String> {
    let base = lib_ollama_path();
    dependency_paths_from_base(&base, library, variant)
        .into_iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect()
}

fn determine_lib_ollama_path<F>(exe_path: &Path, cwd: &Path, exists: F) -> PathBuf
where
    F: Fn(&Path) -> bool,
{
    let exe_dir = exe_path.parent().map(Path::to_path_buf).unwrap_or_default();

    let mut candidates: Vec<PathBuf> = Vec::new();

    let lib_path = if cfg!(target_os = "windows") {
        exe_dir.join("lib").join("ollama")
    } else if cfg!(target_os = "linux") {
        exe_dir.join("..").join("lib").join("ollama")
    } else if cfg!(target_os = "macos") {
        exe_dir.clone()
    } else {
        exe_dir.join("lib").join("ollama")
    };

    candidates.push(lib_path);
    candidates.push(exe_dir.join("build").join("lib").join("ollama"));
    candidates.push(cwd.join("build").join("lib").join("ollama"));

    for path in &candidates {
        if exists(path) {
            return path.clone();
        }
    }

    exe_dir
}

fn dependency_paths_from_base(base: &Path, library: &str, variant: &str) -> Vec<PathBuf> {
    if base.as_os_str().is_empty() {
        return Vec::new();
    }

    let mut paths = Vec::new();

    if !variant.is_empty() {
        let dir = format!("{}_{}", library, variant);
        paths.push(base.join(dir));
    }

    paths.push(base.to_path_buf());
    paths
}

#[cfg(test)]
mod tests {
    use super::{dependency_paths_from_base, determine_lib_ollama_path};
    use std::collections::HashSet;
    use std::path::PathBuf;

    #[test]
    fn selects_platform_specific_directory_when_available() {
        let exe = sample_exe_path();
        let exe_dir = exe.parent().unwrap().to_path_buf();
        let cwd = PathBuf::from("/tmp/workspace");

        let expected = platform_lib_directory(&exe_dir);

        let mut existing = HashSet::new();
        existing.insert(expected.clone());

        let resolved = determine_lib_ollama_path(&exe, &cwd, |p| existing.contains(p));
        assert_eq!(resolved, expected);
    }

    #[test]
    fn prefers_executable_build_directory_when_base_missing() {
        let exe = sample_exe_path();
        let exe_dir = exe.parent().unwrap().to_path_buf();
        let cwd = PathBuf::from("/tmp/workspace");

        let expected = exe_dir.join("build").join("lib").join("ollama");

        let mut existing = HashSet::new();
        existing.insert(expected.clone());

        let resolved = determine_lib_ollama_path(&exe, &cwd, |p| existing.contains(p));
        assert_eq!(resolved, expected);
    }

    #[test]
    fn falls_back_to_cwd_build_directory() {
        let exe = sample_exe_path();
        let cwd = PathBuf::from("/tmp/workspace");

        let expected = cwd.join("build").join("lib").join("ollama");

        let mut existing = HashSet::new();
        existing.insert(expected.clone());

        let resolved = determine_lib_ollama_path(&exe, &cwd, |p| existing.contains(p));
        assert_eq!(resolved, expected);
    }

    #[test]
    fn returns_executable_directory_when_nothing_matches() {
        let exe = sample_exe_path();
        let exe_dir = exe.parent().unwrap().to_path_buf();
        let cwd = PathBuf::from("/tmp/workspace");

        let resolved = determine_lib_ollama_path(&exe, &cwd, |_| false);
        assert_eq!(resolved, exe_dir);
    }

    #[test]
    fn dependency_paths_include_variant_directory() {
        let base = PathBuf::from("/opt/ollama/lib");
        let paths = dependency_paths_from_base(&base, "cuda", "v12");
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0], base.join("cuda_v12"));
        assert_eq!(paths[1], base);
    }

    #[test]
    fn dependency_paths_without_variant_only_include_base() {
        let base = PathBuf::from("/opt/ollama/lib");
        let paths = dependency_paths_from_base(&base, "cpu", "");
        assert_eq!(paths, vec![base]);
    }

    fn sample_exe_path() -> PathBuf {
        if cfg!(target_os = "windows") {
            PathBuf::from(r"C:\\Program Files\\Ollama\\ollama.exe")
        } else {
            PathBuf::from("/opt/ollama/bin/ollama")
        }
    }

    fn platform_lib_directory(exe_dir: &PathBuf) -> PathBuf {
        if cfg!(target_os = "windows") {
            exe_dir.join("lib").join("ollama")
        } else if cfg!(target_os = "linux") {
            exe_dir.join("..").join("lib").join("ollama")
        } else if cfg!(target_os = "macos") {
            exe_dir.clone()
        } else {
            exe_dir.join("lib").join("ollama")
        }
    }
}
