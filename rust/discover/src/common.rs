#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct MemInfo {
    pub total_memory: u64,
    pub free_memory: u64,
    pub free_swap: u64,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct GpuInfo {
    pub mem_info: MemInfo,
    pub library: String,
    pub variant: String,
    pub minimum_memory: u64,
    pub dependency_path: Vec<String>,
    pub env_workarounds: Vec<String>,
    pub unreliable_free_memory: bool,
    pub id: String,
    pub name: String,
    pub compute: String,
    pub driver_major: i32,
    pub driver_minor: i32,
}

impl GpuInfo {
    pub fn runner_name(&self) -> String {
        if self.variant.is_empty() {
            self.library.clone()
        } else {
            format!("{}_{}", self.library, self.variant)
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct CPU {
    pub id: String,
    pub vendor_id: String,
    pub model_name: String,
    pub core_count: i32,
    pub efficiency_core_count: i32,
    pub thread_count: i32,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct CPUInfo {
    pub gpu_info: GpuInfo,
    pub cpus: Vec<CPU>,
}

pub type GpuInfoList = Vec<GpuInfo>;

pub type CudaGPUInfoList = Vec<GpuInfo>;
pub type RocmGPUInfoList = Vec<GpuInfo>;
pub type OneapiGPUInfoList = Vec<GpuInfo>;

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedGPUInfo {
    pub gpu_info: GpuInfo,
    pub reason: String,
}

pub trait GpuInfoListExt {
    fn by_library(&self) -> Vec<GpuInfoList>;
}

impl GpuInfoListExt for GpuInfoList {
    fn by_library(&self) -> Vec<GpuInfoList> {
        let mut resp: Vec<GpuInfoList> = Vec::new();
        let mut libs: Vec<String> = Vec::new();
        for info in self {
            let mut requested = info.library.clone();
            if !info.variant.is_empty() {
                requested.push('_');
                requested.push_str(&info.variant);
            }
            if let Some(pos) = libs.iter().position(|l| l == &requested) {
                resp[pos].push(info.clone());
            } else {
                libs.push(requested);
                resp.push(vec![info.clone()]);
            }
        }
        resp
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct SystemInfo {
    pub system: CPUInfo,
    pub gpus: Vec<GpuInfo>,
    pub unsupported_gpus: Vec<UnsupportedGPUInfo>,
    pub discovery_errors: Vec<String>,
}

impl SystemInfo {
    pub fn get_optimal_thread_count(&self) -> i32 {
        if self.system.cpus.is_empty() {
            return 0;
        }
        let mut core_count = 0;
        for c in &self.system.cpus {
            core_count += c.core_count - c.efficiency_core_count;
        }
        core_count
    }
}
