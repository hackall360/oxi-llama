use crate::{path, GpuInfo, MemInfo, CPU};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use nvml_wrapper::{
    bitmasks::InitFlags, cuda_driver_version_major, cuda_driver_version_minor, error::NvmlError,
    Nvml,
};

#[cfg(feature = "linux-rocm")]
mod hip;

pub fn get_cpu_mem() -> io::Result<MemInfo> {
    let mut mem = MemInfo::default();
    let file = File::open("/proc/meminfo")?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line?;
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let v: u64 = rest
                .trim()
                .split_whitespace()
                .next()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
            mem.total_memory = v * 1024;
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let v: u64 = rest
                .trim()
                .split_whitespace()
                .next()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
            mem.free_memory = v * 1024;
        } else if let Some(rest) = line.strip_prefix("MemFree:") {
            if mem.free_memory == 0 {
                let v: u64 = rest
                    .trim()
                    .split_whitespace()
                    .next()
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0);
                mem.free_memory = v * 1024;
            }
        } else if let Some(rest) = line.strip_prefix("Buffers:") {
            if mem.free_memory == 0 {
                let v: u64 = rest
                    .trim()
                    .split_whitespace()
                    .next()
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0);
                mem.free_memory += v * 1024;
            }
        } else if let Some(rest) = line.strip_prefix("Cached:") {
            if mem.free_memory == 0 {
                let v: u64 = rest
                    .trim()
                    .split_whitespace()
                    .next()
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0);
                mem.free_memory += v * 1024;
            }
        } else if let Some(rest) = line.strip_prefix("SwapFree:") {
            let v: u64 = rest
                .trim()
                .split_whitespace()
                .next()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
            mem.free_swap = v * 1024;
        }
    }
    Ok(mem)
}

pub fn get_gpu_info() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    if let Ok(nvml) = Nvml::init_with_flags(InitFlags::NO_GPUS | InitFlags::NO_ATTACH) {
        if let Ok(mut cuda_gpus) = collect_cuda_info(&nvml) {
            gpus.append(&mut cuda_gpus);
        }
        let _ = nvml.shutdown();
    }

    #[cfg(feature = "linux-rocm")]
    {
        if let Ok(mut hip_gpus) = hip::collect_hip_info() {
            gpus.append(&mut hip_gpus);
        }
    }

    if gpus.is_empty() {
        cpu_fallback()
    } else {
        gpus
    }
}

fn collect_cuda_info(nvml: &Nvml) -> Result<Vec<GpuInfo>, NvmlError> {
    let device_count = nvml.device_count()?;
    if device_count == 0 {
        return Ok(Vec::new());
    }

    let (driver_major, driver_minor, variant) = match nvml.sys_cuda_driver_version() {
        Ok(version) => {
            let major = cuda_driver_version_major(version);
            let minor = cuda_driver_version_minor(version);
            let variant = if major > 0 {
                format!("v{}", major)
            } else {
                String::new()
            };
            (major, minor, variant)
        }
        Err(_) => (0, 0, String::new()),
    };
    let dependency_paths = path::default_dependency_paths("cuda", &variant);

    let mut gpus = Vec::with_capacity(device_count as usize);

    for index in 0..device_count {
        let device = nvml.device_by_index(index)?;
        let memory = device.memory_info()?;

        let mem_info = MemInfo {
            total_memory: memory.total,
            free_memory: memory.free,
            free_swap: 0,
        };

        let id = device.uuid().unwrap_or_else(|_| format!("gpu-{}", index));
        let name = device.name().unwrap_or_else(|_| "NVIDIA GPU".to_string());
        let compute = device
            .cuda_compute_capability()
            .map(|cap| format!("{}.{}", cap.major, cap.minor))
            .unwrap_or_default();

        gpus.push(GpuInfo {
            mem_info,
            library: "cuda".into(),
            variant: variant.clone(),
            dependency_path: dependency_paths.clone(),
            id,
            name,
            compute,
            driver_major,
            driver_minor,
            ..Default::default()
        });
    }

    Ok(gpus)
}

fn cpu_fallback() -> Vec<GpuInfo> {
    let mem = get_cpu_mem().unwrap_or_default();
    let dependency_path = path::default_dependency_paths("cpu", "");
    vec![GpuInfo {
        mem_info: mem,
        library: "cpu".into(),
        dependency_path,
        ..Default::default()
    }]
}

pub fn linux_cpu_details<R: BufRead>(reader: R) -> io::Result<Vec<CPU>> {
    #[derive(Default)]
    struct LinuxCpuInfo {
        id: String,
        vendor_id: String,
        model_name: String,
        physical_id: String,
        siblings: String,
        core_id: String,
    }
    let mut scanner = reader.lines();
    let mut cpu_infos: Vec<LinuxCpuInfo> = Vec::new();
    let mut cpu = LinuxCpuInfo::default();
    while let Some(line) = scanner.next() {
        let line = line?;
        if line.trim().is_empty() {
            if !cpu.id.is_empty() {
                cpu_infos.push(cpu);
                cpu = LinuxCpuInfo::default();
            }
            continue;
        }
        if let Some(idx) = line.find(':') {
            let key = line[..idx].trim();
            let val = line[idx + 1..].trim().to_string();
            match key {
                "processor" => cpu.id = val,
                "vendor_id" => cpu.vendor_id = val,
                "model name" => cpu.model_name = val,
                "physical id" => cpu.physical_id = val,
                "siblings" => cpu.siblings = val,
                "core id" => cpu.core_id = val,
                _ => {}
            }
        }
    }
    if !cpu.id.is_empty() {
        cpu_infos.push(cpu);
    }

    let mut socket_by_id: HashMap<String, CPU> = HashMap::new();
    let mut core_by_socket: HashMap<String, HashSet<String>> = HashMap::new();
    let mut threads_by_core_by_socket: HashMap<String, HashMap<String, i32>> = HashMap::new();
    for c in cpu_infos {
        let socket = c.physical_id.clone();
        socket_by_id.entry(socket.clone()).or_insert_with(|| CPU {
            id: c.physical_id.clone(),
            vendor_id: c.vendor_id.clone(),
            model_name: c.model_name.clone(),
            core_count: 0,
            efficiency_core_count: 0,
            thread_count: 0,
        });
        core_by_socket.entry(socket.clone()).or_default();
        threads_by_core_by_socket.entry(socket.clone()).or_default();
        let core_key = if !c.core_id.is_empty() {
            format!("{}:{}", socket, c.core_id)
        } else {
            format!("{}:{}", socket, c.id)
        };
        core_by_socket
            .get_mut(&socket)
            .unwrap()
            .insert(core_key.clone());
        *threads_by_core_by_socket
            .get_mut(&socket)
            .unwrap()
            .entry(core_key)
            .or_insert(0) += 1;
    }

    for (id, cpu) in socket_by_id.iter_mut() {
        cpu.core_count = core_by_socket[id].len() as i32;
        let mut efficiency = 0;
        let mut total_threads = 0;
        for threads in threads_by_core_by_socket[id].values() {
            total_threads += *threads;
            if *threads == 1 {
                efficiency += 1;
            }
        }
        cpu.thread_count = total_threads as i32;
        cpu.efficiency_core_count = if efficiency == cpu.core_count {
            0
        } else {
            efficiency
        };
    }

    let mut keys: Vec<String> = socket_by_id.keys().cloned().collect();
    keys.sort();
    let mut result = Vec::new();
    for k in keys {
        if let Some(cpu) = socket_by_id.remove(&k) {
            result.push(cpu);
        }
    }
    Ok(result)
}

pub fn get_cpu_details() -> io::Result<Vec<CPU>> {
    let file = File::open(Path::new("/proc/cpuinfo"))?;
    linux_cpu_details(BufReader::new(file))
}
