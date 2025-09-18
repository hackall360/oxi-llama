use std::sync::Arc;

use discover::{GpuInfo, GpuInfoList, MemInfo};
use llm::*;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

#[test]
fn test_llm_server_fit_gpu() {
    struct Gpu {
        library: &'static str,
        free: u64,
    }
    struct TestCase {
        name: &'static str,
        gpus: Vec<Gpu>,
        layers: Vec<u64>,
        num_gpu: i32,
        require_full: bool,
        expected: GpuLayersList,
        expected_err: Option<LlmError>,
    }
    fn gl(id: &str, layers: &[usize]) -> GpuLayers {
        GpuLayers {
            id: id.to_string(),
            layers: layers.to_vec(),
        }
    }
    let tests = vec![
        TestCase {
            name: "No GPU",
            gpus: vec![],
            layers: vec![50 * MEBIBYTE, 50 * MEBIBYTE, 50 * MEBIBYTE],
            num_gpu: -1,
            require_full: false,
            expected: vec![],
            expected_err: None,
        },
        TestCase {
            name: "Full single GPU",
            gpus: vec![Gpu {
                library: "",
                free: 256 * MEBIBYTE,
            }],
            layers: vec![50 * MEBIBYTE, 50 * MEBIBYTE, 50 * MEBIBYTE],
            num_gpu: -1,
            require_full: false,
            expected: vec![gl("gpu0", &[0, 1, 2])],
            expected_err: None,
        },
        TestCase {
            name: "Partial single GPU",
            gpus: vec![Gpu {
                library: "",
                free: 256 * MEBIBYTE,
            }],
            layers: vec![
                100 * MEBIBYTE,
                100 * MEBIBYTE,
                100 * MEBIBYTE,
                100 * MEBIBYTE,
            ],
            num_gpu: -1,
            require_full: false,
            expected: vec![gl("gpu0", &[1, 2])],
            expected_err: None,
        },
        TestCase {
            name: "Single GPU with numGPU 1",
            gpus: vec![Gpu {
                library: "",
                free: 256 * MEBIBYTE,
            }],
            layers: vec![50 * MEBIBYTE, 50 * MEBIBYTE, 50 * MEBIBYTE],
            num_gpu: 1,
            require_full: false,
            expected: vec![gl("gpu0", &[1])],
            expected_err: None,
        },
        TestCase {
            name: "Single GPU with numGPU 0",
            gpus: vec![Gpu {
                library: "",
                free: 256 * MEBIBYTE,
            }],
            layers: vec![50 * MEBIBYTE, 50 * MEBIBYTE, 50 * MEBIBYTE],
            num_gpu: 0,
            require_full: false,
            expected: vec![],
            expected_err: None,
        },
        TestCase {
            name: "Single GPU with numGPU 999",
            gpus: vec![Gpu {
                library: "",
                free: 256 * MEBIBYTE,
            }],
            layers: vec![
                100 * MEBIBYTE,
                100 * MEBIBYTE,
                100 * MEBIBYTE,
                100 * MEBIBYTE,
            ],
            num_gpu: 999,
            require_full: false,
            expected: vec![gl("gpu0", &[0, 1, 2, 3])],
            expected_err: None,
        },
        TestCase {
            name: "Multi GPU fits on one",
            gpus: vec![
                Gpu {
                    library: "",
                    free: 128 * MEBIBYTE,
                },
                Gpu {
                    library: "",
                    free: 256 * MEBIBYTE,
                },
            ],
            layers: vec![50 * MEBIBYTE, 50 * MEBIBYTE, 50 * MEBIBYTE],
            num_gpu: -1,
            require_full: false,
            expected: vec![gl("gpu1", &[0, 1, 2])],
            expected_err: None,
        },
        TestCase {
            name: "Multi GPU split",
            gpus: vec![
                Gpu {
                    library: "",
                    free: 128 * MEBIBYTE,
                },
                Gpu {
                    library: "",
                    free: 256 * MEBIBYTE,
                },
            ],
            layers: vec![256 * MEBIBYTE, 50 * MEBIBYTE, 50 * MEBIBYTE],
            num_gpu: -1,
            require_full: false,
            expected: vec![gl("gpu1", &[0]), gl("gpu0", &[1, 2])],
            expected_err: None,
        },
        TestCase {
            name: "Multi GPU partial",
            gpus: vec![
                Gpu {
                    library: "",
                    free: 128 * MEBIBYTE,
                },
                Gpu {
                    library: "",
                    free: 256 * MEBIBYTE,
                },
            ],
            layers: vec![256 * MEBIBYTE, 256 * MEBIBYTE, 50 * MEBIBYTE],
            num_gpu: -1,
            require_full: false,
            expected: vec![gl("gpu1", &[1])],
            expected_err: None,
        },
        TestCase {
            name: "Multi GPU numGPU 1",
            gpus: vec![
                Gpu {
                    library: "",
                    free: 128 * MEBIBYTE,
                },
                Gpu {
                    library: "",
                    free: 256 * MEBIBYTE,
                },
            ],
            layers: vec![50 * MEBIBYTE, 50 * MEBIBYTE, 50 * MEBIBYTE],
            num_gpu: 1,
            require_full: false,
            expected: vec![gl("gpu1", &[1])],
            expected_err: None,
        },
        TestCase {
            name: "Multi GPU numGPU 2",
            gpus: vec![
                Gpu {
                    library: "",
                    free: 128 * MEBIBYTE,
                },
                Gpu {
                    library: "",
                    free: 256 * MEBIBYTE,
                },
            ],
            layers: vec![256 * MEBIBYTE, 50 * MEBIBYTE, 50 * MEBIBYTE],
            num_gpu: 2,
            require_full: false,
            expected: vec![gl("gpu1", &[0]), gl("gpu0", &[1])],
            expected_err: None,
        },
        TestCase {
            name: "Multi GPU numGPU 999",
            gpus: vec![
                Gpu {
                    library: "",
                    free: 128 * MEBIBYTE,
                },
                Gpu {
                    library: "",
                    free: 256 * MEBIBYTE,
                },
            ],
            layers: vec![256 * MEBIBYTE, 256 * MEBIBYTE, 50 * MEBIBYTE],
            num_gpu: 999,
            require_full: false,
            expected: vec![gl("gpu1", &[0, 1]), gl("gpu0", &[2])],
            expected_err: None,
        },
        TestCase {
            name: "Multi GPU different libraries",
            gpus: vec![
                Gpu {
                    library: "cuda",
                    free: 128 * MEBIBYTE,
                },
                Gpu {
                    library: "rocm",
                    free: 256 * MEBIBYTE,
                },
            ],
            layers: vec![128 * MEBIBYTE, 128 * MEBIBYTE, 50 * MEBIBYTE],
            num_gpu: -1,
            require_full: false,
            expected: vec![gl("gpu1", &[0, 1])],
            expected_err: None,
        },
        TestCase {
            name: "requireFull",
            gpus: vec![Gpu {
                library: "",
                free: 256 * MEBIBYTE,
            }],
            layers: vec![
                100 * MEBIBYTE,
                100 * MEBIBYTE,
                100 * MEBIBYTE,
                100 * MEBIBYTE,
            ],
            num_gpu: -1,
            require_full: true,
            expected: vec![],
            expected_err: Some(LlmError::LoadRequiredFull),
        },
    ];

    for tt in tests {
        let system_info = SystemInfo {
            system: SystemMemory {
                total_memory: GIBIBYTE,
                free_memory: 512 * MEBIBYTE,
                free_swap: 256 * MEBIBYTE,
            },
        };
        let mut gpus: GpuInfoList = Vec::new();
        for (i, g) in tt.gpus.iter().enumerate() {
            gpus.push(GpuInfo {
                id: format!("gpu{}", i),
                library: g.library.to_string(),
                mem_info: MemInfo {
                    total_memory: g.free,
                    free_memory: g.free,
                    free_swap: 0,
                },
                minimum_memory: 0,
                ..Default::default()
            });
        }
        let mut s = OllamaServer {
            llm: LlmServer {
                total_layers: tt.layers.len() as u64,
                options: Options {
                    runner: RunnerOptions {
                        num_gpu: tt.num_gpu,
                    },
                    ..Default::default()
                },
                sem: Arc::new(Semaphore::new(1)),
            },
            mem: None,
        };
        let mut mem = BackendMemory {
            cpu: DeviceMemory {
                weights: vec![Memory::default(); s.llm.total_layers as usize],
                cache: vec![Memory::default(); s.llm.total_layers as usize],
                ..Default::default()
            },
            gpus: vec![],
            ..Default::default()
        };
        for (i, size) in tt.layers.iter().enumerate() {
            mem.cpu.weights[i].size = *size;
        }
        for (i, _) in gpus.iter().enumerate() {
            mem.gpus.push(DeviceMemory {
                id: format!("gpu{}", i),
                weights: vec![Memory::default(); s.llm.total_layers as usize],
                cache: vec![Memory::default(); s.llm.total_layers as usize],
                ..Default::default()
            });
        }
        s.mem = Some(mem.clone());
        let result = s.create_layout(&system_info, &gpus, s.mem.as_ref(), tt.require_full, 0.0);
        match (result, tt.expected_err) {
            (Err(e), Some(exp)) => assert_eq!(e, exp, "{}", tt.name),
            (Err(e), None) => panic!("{}: unexpected error {:?}", tt.name, e),
            (Ok(got), Some(exp)) => {
                panic!("{}: expected error {:?} but got {:?}", tt.name, exp, got)
            }
            (Ok(got), None) => {
                assert_eq!(got.hash_value(), tt.expected.hash_value(), "{}", tt.name)
            }
        }
    }
}

#[tokio::test]
async fn test_llm_server_completion_format() {
    let token = CancellationToken::new();
    let s = OllamaServer {
        llm: LlmServer {
            sem: Arc::new(Semaphore::new(1)),
            ..Default::default()
        },
        ..Default::default()
    };
    async fn check_invalid(s: &OllamaServer, token: &CancellationToken, format: &str) {
        let req = CompletionRequest {
            format: Some(format.as_bytes().to_vec()),
            options: Some(Options::default()),
        };
        match s.completion(token, req).await {
            Err(LlmError::InvalidFormat(_)) => {}
            other => panic!("expected invalid format, got {:?}", other),
        }
    }
    check_invalid(&s, &token, "X").await;
    check_invalid(&s, &token, "\"X\"").await;
    token.cancel();
    async fn check_valid(s: &OllamaServer, token: &CancellationToken, format: Option<&str>) {
        let req = CompletionRequest {
            format: format.map(|f| f.as_bytes().to_vec()),
            options: Some(Options::default()),
        };
        match s.completion(token, req).await {
            Err(LlmError::Cancelled) => {}
            other => panic!("expected cancelled, got {:?}", other),
        }
    }
    let valids = ["", "\"\"", "null", "\"json\"", "{\"type\":\"object\"}"];
    for v in valids {
        check_valid(&s, &token, Some(v)).await;
    }
    check_valid(&s, &token, None).await;
}
