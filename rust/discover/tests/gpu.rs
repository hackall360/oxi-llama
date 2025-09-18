use discover::{get_cpu_mem, get_gpu_info, GpuInfo, GpuInfoListExt};

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
    let test_cases = vec![
        (vec![], 0),
        (
            vec![GpuInfo {
                library: "cpu".into(),
                ..Default::default()
            }],
            1,
        ),
        (
            vec![
                GpuInfo {
                    library: "cpu".into(),
                    ..Default::default()
                },
                GpuInfo {
                    library: "cuda".into(),
                    ..Default::default()
                },
            ],
            2,
        ),
        (
            vec![
                GpuInfo {
                    library: "cpu".into(),
                    ..Default::default()
                },
                GpuInfo {
                    library: "cuda".into(),
                    ..Default::default()
                },
                GpuInfo {
                    library: "cuda".into(),
                    ..Default::default()
                },
            ],
            2,
        ),
        (
            vec![
                GpuInfo {
                    library: "cpu".into(),
                    ..Default::default()
                },
                GpuInfo {
                    library: "cuda".into(),
                    variant: "v11".into(),
                    ..Default::default()
                },
                GpuInfo {
                    library: "cuda".into(),
                    variant: "v11".into(),
                    ..Default::default()
                },
            ],
            2,
        ),
        (
            vec![
                GpuInfo {
                    library: "cpu".into(),
                    ..Default::default()
                },
                GpuInfo {
                    library: "cuda".into(),
                    variant: "v11".into(),
                    ..Default::default()
                },
                GpuInfo {
                    library: "cuda".into(),
                    variant: "v12".into(),
                    ..Default::default()
                },
            ],
            3,
        ),
    ];

    for (input, expect) in test_cases {
        let list = input.by_library();
        assert_eq!(list.len(), expect);
    }
}
