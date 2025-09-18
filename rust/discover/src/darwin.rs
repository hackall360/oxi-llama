use crate::{GpuInfo, MemInfo};
use sysinfo::SystemExt;

pub fn get_cpu_mem() -> std::io::Result<MemInfo> {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    Ok(MemInfo {
        total_memory: sys.total_memory(),
        free_memory: sys.available_memory(),
        free_swap: sys.free_swap(),
    })
}

pub fn get_gpu_info() -> Vec<GpuInfo> {
    let mem = get_cpu_mem().unwrap_or_default();
    vec![GpuInfo {
        mem_info: mem,
        library: "cpu".into(),
        ..Default::default()
    }]
}
