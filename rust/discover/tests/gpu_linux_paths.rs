#![cfg(target_os = "linux")]

#[cfg(feature = "linux-rocm")]
use discover::test_support::build_rocm_info;
use discover::test_support::{
    build_cuda_info, build_oneapi_info, override_cuda, override_hip, override_oneapi, MockDevice,
};
use discover::{get_gpu_info, path, MemInfo};

#[test]
fn test_cuda_mock_devices_populate_gpu_info() {
    let base_mem = MemInfo {
        total_memory: 16 * 1024,
        free_memory: 8 * 1024,
        free_swap: 0,
    };
    let devices = vec![MockDevice::new(
        "gpu-0",
        "Test CUDA",
        base_mem.clone(),
        "8.0",
        0,
        0,
        "",
    )];

    let info = build_cuda_info(devices, 12, 4);
    assert_eq!(info.len(), 1);
    let gpu = &info[0];
    assert_eq!(gpu.library, "cuda");
    assert_eq!(gpu.variant, "v12");
    assert_eq!(gpu.id, "gpu-0");
    assert_eq!(gpu.name, "Test CUDA");
    assert_eq!(gpu.compute, "8.0");
    assert_eq!(gpu.driver_major, 12);
    assert_eq!(gpu.driver_minor, 4);
    assert_eq!(gpu.mem_info, base_mem);
    assert_eq!(
        gpu.dependency_path,
        path::default_dependency_paths("cuda", "v12"),
    );
}

#[cfg(feature = "linux-rocm")]
#[test]
fn test_rocm_mock_devices_populate_gpu_info() {
    let base_mem = MemInfo {
        total_memory: 32 * 1024,
        free_memory: 12 * 1024,
        free_swap: 0,
    };
    let devices = vec![MockDevice::new(
        "hip-0",
        "Test ROCm",
        base_mem.clone(),
        "9.0",
        0,
        0,
        "",
    )];

    let info = build_rocm_info(devices, 5, 1, Some(6));
    assert_eq!(info.len(), 1);
    let gpu = &info[0];
    assert_eq!(gpu.library, "rocm");
    assert_eq!(gpu.variant, "v6");
    assert_eq!(gpu.driver_major, 5);
    assert_eq!(gpu.driver_minor, 1);
    assert_eq!(gpu.mem_info, base_mem);
    assert_eq!(
        gpu.dependency_path,
        path::default_dependency_paths("rocm", "v6"),
    );
}

#[test]
fn test_oneapi_mock_devices_populate_gpu_info() {
    let base_mem = MemInfo {
        total_memory: 24 * 1024,
        free_memory: 10 * 1024,
        free_swap: 0,
    };
    let devices = vec![MockDevice::new(
        "oneapi-0",
        "Level Zero",
        base_mem.clone(),
        String::new(),
        7,
        3,
        "v7",
    )];

    let info = build_oneapi_info(devices);
    assert_eq!(info.len(), 1);
    let gpu = &info[0];
    assert_eq!(gpu.library, "oneapi");
    assert_eq!(gpu.variant, "v7");
    assert_eq!(gpu.driver_major, 7);
    assert_eq!(gpu.driver_minor, 3);
    assert_eq!(gpu.mem_info, base_mem);
    assert_eq!(
        gpu.dependency_path,
        path::default_dependency_paths("oneapi", "v7"),
    );
}

#[test]
fn test_cpu_fallback_when_no_gpu_libraries() {
    let _cuda_guard = override_cuda(Some(Vec::new()));
    let _hip_guard = override_hip(Some(Vec::new()));
    let _oneapi_guard = override_oneapi(Some(Vec::new()));

    let gpus = get_gpu_info();
    assert_eq!(gpus.len(), 1);
    assert_eq!(gpus[0].library, "cpu");
}
