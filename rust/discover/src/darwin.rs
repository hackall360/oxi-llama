use crate::{GpuInfo, MemInfo};
use metal::Device;
use objc::rc::autoreleasepool;
use std::panic::{self, AssertUnwindSafe};
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
    let cpu_mem = get_cpu_mem().unwrap_or_default();

    let metal_devices = panic::catch_unwind(AssertUnwindSafe(|| {
        autoreleasepool(|| {
            let dependency_paths = crate::path::default_dependency_paths("metal", "");
            collect_metal_devices(&dependency_paths)
        })
    }));

    if let Ok(gpus) = metal_devices {
        if !gpus.is_empty() {
            return gpus;
        }
    }

    cpu_fallback_with_mem(cpu_mem)
}

fn collect_metal_devices(dependency_paths: &[String]) -> Vec<GpuInfo> {
    let mut devices = Device::all();
    if devices.is_empty() {
        if let Some(device) = Device::system_default() {
            devices.push(device);
        }
    }

    let mut infos = Vec::with_capacity(devices.len());

    for device in devices {
        infos.push(build_gpu_info(device, dependency_paths));
    }

    infos
}

fn build_gpu_info(device: Device, dependency_paths: &[String]) -> GpuInfo {
    let recommended = device.recommended_max_working_set_size();
    let allocated = device.current_allocated_size();

    let free_memory = recommended.saturating_sub(allocated);

    let mut variant_tags = Vec::new();
    if device.is_low_power() {
        variant_tags.push("low_power");
    }
    if device.is_headless() {
        variant_tags.push("headless");
    }
    if device.is_removable() {
        variant_tags.push("removable");
    }
    let variant = variant_tags.join("_");

    GpuInfo {
        mem_info: MemInfo {
            total_memory: recommended,
            free_memory,
            free_swap: 0,
        },
        library: "metal".into(),
        variant,
        minimum_memory: recommended,
        dependency_path: dependency_paths.to_vec(),
        unreliable_free_memory: recommended == 0,
        id: format!("metal-0x{:x}", device.registry_id()),
        name: device.name().to_string(),
        ..Default::default()
    }
}

fn cpu_fallback_with_mem(mem: MemInfo) -> Vec<GpuInfo> {
    let dependency_path = crate::path::default_dependency_paths("cpu", "");
    vec![GpuInfo {
        mem_info: mem,
        library: "cpu".into(),
        dependency_path,
        ..Default::default()
    }]
}
