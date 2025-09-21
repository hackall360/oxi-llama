use discover::{get_cpu_mem, get_gpu_info, GpuInfo, GpuInfoListExt, MemInfo};
use std::collections::HashSet;

#[cfg(target_os = "linux")]
use discover::test_support::{build_cuda_info, build_oneapi_info, build_rocm_info, MockDevice};

#[cfg(target_os = "macos")]
use discover::path::{default_dependency_paths, lib_ollama_path};

#[cfg(target_os = "macos")]
use discover::collect_metal_devices_for_tests;

fn fixture_dependency_paths(library: &str, variant: &str) -> Vec<String> {
    let base = format!("/opt/ollama/{library}/base");
    if variant.is_empty() {
        vec![base]
    } else {
        vec![format!("/opt/ollama/{library}/{library}_{variant}"), base]
    }
}

fn fixture_mem(total_mb: u64, free_mb: u64) -> MemInfo {
    MemInfo {
        total_memory: total_mb * 1024,
        free_memory: free_mb * 1024,
        free_swap: 0,
    }
}

fn fixture_gpu(
    library: &str,
    variant: &str,
    id_suffix: &str,
    total_mb: u64,
    free_mb: u64,
) -> GpuInfo {
    let mem_info = fixture_mem(total_mb, free_mb);
    GpuInfo {
        mem_info,
        library: library.into(),
        variant: variant.into(),
        minimum_memory: total_mb * 1024,
        dependency_path: fixture_dependency_paths(library, variant),
        id: format!("{library}-{id_suffix}"),
        name: format!("{} {}", library.to_uppercase(), id_suffix),
        compute: match library {
            "cuda" => "8.9".into(),
            "rocm" => "9.0".into(),
            "metal" => "3.0".into(),
            "oneapi" => "1.0".into(),
            _ => String::new(),
        },
        driver_major: match library {
            "cuda" => 12,
            "rocm" => 6,
            "metal" => 3,
            "oneapi" => 1,
            _ => 0,
        },
        driver_minor: match library {
            "cuda" => 4,
            "rocm" => 1,
            "metal" => 0,
            "oneapi" => 0,
            _ => 0,
        },
        ..Default::default()
    }
}

fn fixture_gpu_inventory() -> Vec<GpuInfo> {
    vec![
        fixture_gpu("cpu", "", "0", 16 * 1024, 12 * 1024),
        fixture_gpu("cuda", "v12", "0", 48 * 1024, 40 * 1024),
        fixture_gpu("cuda", "v12", "1", 48 * 1024, 39 * 1024),
        fixture_gpu("cuda", "v11", "legacy", 32 * 1024, 24 * 1024),
        fixture_gpu("rocm", "v6", "0", 64 * 1024, 50 * 1024),
        fixture_gpu("metal", "low_power", "0", 32 * 1024, 28 * 1024),
        fixture_gpu("oneapi", "level_zero", "0", 24 * 1024, 20 * 1024),
    ]
}

#[test]
fn test_basic_get_gpu_info() {
    let info = get_gpu_info();
    assert!(!info.is_empty());
    let lib = &info[0].library;
    assert!(matches!(lib.as_str(), "cuda" | "rocm" | "cpu" | "metal"));
    if lib != "cpu" {
        assert!(info[0].mem_info.total_memory > 0);
        assert!(info[0].mem_info.free_memory > 0);
    }
}

#[test]
fn test_cpu_mem_info() {
    let info = get_cpu_mem().expect("cpu mem");
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        assert!(info.total_memory > 0);
        assert!(info.free_memory > 0);
    }
}

#[test]
fn test_by_library() {
    let list = fixture_gpu_inventory().by_library();
    let summary: Vec<(String, String, usize)> = list
        .into_iter()
        .map(|group| {
            let library = group[0].library.clone();
            let variant = group[0].variant.clone();
            let count = group.len();
            (library, variant, count)
        })
        .collect();

    assert_eq!(
        summary,
        vec![
            ("cpu".into(), String::new(), 1),
            ("cuda".into(), "v12".into(), 2),
            ("cuda".into(), "v11".into(), 1),
            ("rocm".into(), "v6".into(), 1),
            ("metal".into(), "low_power".into(), 1),
            ("oneapi".into(), "level_zero".into(), 1),
        ]
    );
}

#[test]
fn test_fixture_covers_accelerator_libraries() {
    let gpus = fixture_gpu_inventory();
    let libraries: HashSet<&str> = gpus.iter().map(|gpu| gpu.library.as_str()).collect();

    for required in ["cuda", "rocm", "metal", "oneapi"] {
        assert!(libraries.contains(required), "missing {required} fixture");
    }
}

#[cfg(target_os = "linux")]
fn linux_mock_device(
    id: &str,
    name: &str,
    compute: &str,
    total_mb: u64,
    free_mb: u64,
    variant: &str,
) -> MockDevice {
    MockDevice::new(
        id,
        name,
        fixture_mem(total_mb, free_mb),
        compute,
        0,
        0,
        variant,
    )
}

#[cfg(target_os = "linux")]
#[test]
fn test_linux_mock_builders_produce_expected_variants() {
    let cuda_devices = vec![
        linux_mock_device("cuda-0", "Ada 0", "9.0", 48 * 1024, 40 * 1024, ""),
        linux_mock_device("cuda-1", "Ada 1", "9.0", 48 * 1024, 39 * 1024, ""),
    ];
    let cuda_info = build_cuda_info(cuda_devices, 12, 4);
    assert_eq!(cuda_info.len(), 2);
    assert!(cuda_info.iter().all(|gpu| gpu.library == "cuda"));
    assert!(cuda_info.iter().all(|gpu| gpu.variant == "v12"));
    assert!(cuda_info.iter().all(|gpu| !gpu.dependency_path.is_empty()));

    let rocm_devices = vec![linux_mock_device(
        "rocm-0",
        "MI300",
        "9.0",
        64 * 1024,
        52 * 1024,
        "",
    )];
    let rocm_info = build_rocm_info(rocm_devices, 6, 1, Some(6));
    assert_eq!(rocm_info.len(), 1);
    assert_eq!(rocm_info[0].library, "rocm");
    assert_eq!(rocm_info[0].variant, "v6");

    let oneapi_devices = vec![linux_mock_device(
        "oneapi-0",
        "Arc",
        "1.0",
        24 * 1024,
        21 * 1024,
        "level_zero",
    )];
    let oneapi_info = build_oneapi_info(oneapi_devices);
    assert_eq!(oneapi_info.len(), 1);
    assert_eq!(oneapi_info[0].library, "oneapi");
    assert_eq!(oneapi_info[0].variant, "level_zero");
}

#[cfg(target_os = "macos")]
#[test]
fn test_metal_dependency_paths_variant_precedes_base() {
    let paths = default_dependency_paths("metal", "low_power");
    assert!(paths.len() >= 2, "expected variant and base paths");
    assert!(
        paths[0].contains("metal_low_power"),
        "variant path should be first: {:?}",
        paths
    );

    let base = lib_ollama_path().to_string_lossy().into_owned();
    assert_eq!(paths.last().unwrap(), &base);
}

#[cfg(target_os = "macos")]
#[test]
fn test_collect_metal_devices_respects_dependency_paths() {
    let dependency_paths = vec![
        "/tmp/ollama/metal_variant".to_string(),
        "/tmp/ollama/metal_base".to_string(),
    ];
    let gpus = collect_metal_devices_for_tests(&dependency_paths);
    assert!(!gpus.is_empty(), "expected at least one Metal device");
    for gpu in gpus {
        assert_eq!(gpu.library, "metal");
        assert_eq!(gpu.dependency_path, dependency_paths);
    }
}
