#[cfg(target_os = "windows")]
use base64::{engine::general_purpose, Engine as _};
#[cfg(target_os = "windows")]
use discover::process_system_logical_processor_information_list;
#[cfg(target_os = "windows")]
use discover::{get_gpu_info_with, GpuInfo, GpuLoaders, MemInfo};

#[cfg(target_os = "windows")]
fn mock_gpu(library: &str, variant: &str, id: &str) -> GpuInfo {
    GpuInfo {
        mem_info: MemInfo {
            total_memory: 16 * 1024,
            free_memory: 8 * 1024,
            free_swap: 0,
        },
        library: library.to_string(),
        variant: variant.to_string(),
        dependency_path: vec![format!("{library}-deps")],
        id: id.to_string(),
        name: format!("{} device", library.to_uppercase()),
        compute: "1.0".into(),
        driver_major: 1,
        driver_minor: 2,
        ..Default::default()
    }
}

#[cfg(target_os = "windows")]
#[test]
fn test_process_system_logical_processor_information_list() {
    struct Pkgs {
        cores: i32,
        efficiency: i32,
        threads: i32,
    }
    let test_cases: &[(&str, &str, &[Pkgs])] = &[
        (
            "AMD64 Family 25 Model 97 Stepping 2 ",
            include_str!("data/case1.b64"),
            &[Pkgs {
                cores: 16,
                efficiency: 0,
                threads: 32,
            }],
        ),
        (
            "Intel64 Family 6 Model 183 Stepping 1 ",
            include_str!("data/case2.b64"),
            &[Pkgs {
                cores: 16,
                efficiency: 8,
                threads: 24,
            }],
        ),
        (
            "dual Intel64 Family 6 Model 85 Stepping 4 ",
            include_str!("data/case3.b64"),
            &[
                Pkgs {
                    cores: 40,
                    efficiency: 0,
                    threads: 80,
                },
                Pkgs {
                    cores: 40,
                    efficiency: 0,
                    threads: 80,
                },
            ],
        ),
    ];
    for (name, b64, expected) in test_cases {
        let raw = general_purpose::STANDARD
            .decode(b64.lines().collect::<String>())
            .expect("valid base64");
        let resp = process_system_logical_processor_information_list(&raw);
        assert_eq!(resp.len(), expected.len(), "{name}: package count");
        for (i, pkg) in expected.iter().enumerate() {
            assert_eq!(resp[i].core_count, pkg.cores, "[{name}] cores");
            assert_eq!(
                resp[i].efficiency_core_count, pkg.efficiency,
                "[{name}] efficiency"
            );
            assert_eq!(resp[i].thread_count, pkg.threads, "[{name}] threads");
        }
    }
}

#[cfg(target_os = "windows")]
#[test]
fn test_get_gpu_info_with_mock_cuda() {
    let expected = mock_gpu("cuda", "v12", "cuda-0");
    let cuda_loader = {
        let gpu = expected.clone();
        move || Ok(vec![gpu.clone()])
    };
    let failing_loader = || -> Result<Vec<GpuInfo>, String> { Err("missing".into()) };

    let loaders = GpuLoaders::new(&cuda_loader, &failing_loader, &failing_loader);
    let info = get_gpu_info_with(&loaders);

    assert_eq!(info, vec![expected]);
}

#[cfg(target_os = "windows")]
#[test]
fn test_get_gpu_info_with_mock_hip() {
    let expected = mock_gpu("rocm", "v6", "hip-0");
    let hip_loader = {
        let gpu = expected.clone();
        move || Ok(vec![gpu.clone()])
    };
    let failing_loader = || -> Result<Vec<GpuInfo>, String> { Err("missing".into()) };

    let loaders = GpuLoaders::new(&failing_loader, &hip_loader, &failing_loader);
    let info = get_gpu_info_with(&loaders);

    assert_eq!(info, vec![expected]);
}

#[cfg(target_os = "windows")]
#[test]
fn test_get_gpu_info_with_mock_oneapi() {
    let expected = mock_gpu("oneapi", "v1", "oneapi-0");
    let oneapi_loader = {
        let gpu = expected.clone();
        move || Ok(vec![gpu.clone()])
    };
    let failing_loader = || -> Result<Vec<GpuInfo>, String> { Err("missing".into()) };

    let loaders = GpuLoaders::new(&failing_loader, &failing_loader, &oneapi_loader);
    let info = get_gpu_info_with(&loaders);

    assert_eq!(info, vec![expected]);
}

#[cfg(target_os = "windows")]
#[test]
fn test_get_gpu_info_fallback_on_failures() {
    let failing_loader = || -> Result<Vec<GpuInfo>, String> { Err("missing".into()) };
    let loaders = GpuLoaders::new(&failing_loader, &failing_loader, &failing_loader);

    let info = get_gpu_info_with(&loaders);

    assert_eq!(info.len(), 1);
    assert_eq!(info[0].library, "cpu");
}
