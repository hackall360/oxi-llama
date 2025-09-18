#[cfg(target_os = "windows")]
use base64::{engine::general_purpose, Engine as _};
#[cfg(target_os = "windows")]
use discover::process_system_logical_processor_information_list;

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
