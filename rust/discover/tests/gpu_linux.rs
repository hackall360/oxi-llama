#[cfg(target_os = "linux")]
use discover::{linux_cpu_details, SystemInfo, CPUInfo};
#[cfg(target_os = "linux")]
use std::io::BufReader;

#[cfg(target_os = "linux")]
#[test]
fn test_linux_cpu_details() {
    let input = r"processor   : 0
vendor_id   : AuthenticAMD
model name  : Dummy
physical id : 0
core id     : 0

processor   : 1
vendor_id   : AuthenticAMD
model name  : Dummy
physical id : 0
core id     : 0

processor   : 2
vendor_id   : AuthenticAMD
model name  : Dummy
physical id : 0
core id     : 1

processor   : 3
vendor_id   : AuthenticAMD
model name  : Dummy
physical id : 0
core id     : 1

processor   : 4
vendor_id   : AuthenticAMD
model name  : Dummy
physical id : 1
core id     : 0

processor   : 5
vendor_id   : AuthenticAMD
model name  : Dummy
physical id : 1
core id     : 0

processor   : 6
vendor_id   : AuthenticAMD
model name  : Dummy
physical id : 1
core id     : 1

processor   : 7
vendor_id   : AuthenticAMD
model name  : Dummy
physical id : 1
core id     : 1
";
    let buf = BufReader::new(input.as_bytes());
    let cpus = linux_cpu_details(buf).expect("parse");
    assert_eq!(cpus.len(), 2);
    assert_eq!(cpus[0].core_count, 2);
    assert_eq!(cpus[0].efficiency_core_count, 0);
    assert_eq!(cpus[0].thread_count, 4);
    assert_eq!(cpus[1].core_count, 2);
    assert_eq!(cpus[1].efficiency_core_count, 0);
    assert_eq!(cpus[1].thread_count, 4);
    let si = SystemInfo { system: CPUInfo { cpus, ..Default::default() }, ..Default::default() };
    assert_eq!(si.get_optimal_thread_count(), 4);
}
