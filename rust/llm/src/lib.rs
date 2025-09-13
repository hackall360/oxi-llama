use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use discover::{GpuInfoList, GpuInfoListExt};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

mod process;
pub use process::{filtered_env, spawn_llama_server};
mod status;
pub use status::StatusWriter;

pub const MEBIBYTE: u64 = 1024 * 1024;
pub const GIBIBYTE: u64 = 1024 * 1024 * 1024;

#[derive(Clone, Default, Debug)]
pub struct Memory {
    pub size: u64,
}

#[derive(Clone, Default, Debug)]
pub struct DeviceMemory {
    pub id: String,
    pub graph: Memory,
    pub weights: Vec<Memory>,
    pub cache: Vec<Memory>,
}

impl DeviceMemory {
    pub fn allocated(&self) -> u64 {
        self.graph.size
            + self.weights.iter().map(|m| m.size).sum::<u64>()
            + self.cache.iter().map(|m| m.size).sum::<u64>()
    }
}

#[derive(Clone, Default, Debug)]
pub struct BackendMemory {
    pub input_weights: Memory,
    pub cpu: DeviceMemory,
    pub gpus: Vec<DeviceMemory>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuLayers {
    pub id: String,
    pub layers: Vec<usize>,
}

pub type GpuLayersList = Vec<GpuLayers>;

pub trait GpuLayersListExt {
    fn sum(&self) -> usize;
    fn hash_value(&self) -> u64;
}

impl GpuLayersListExt for GpuLayersList {
    fn sum(&self) -> usize {
        self.iter().map(|g| g.layers.len()).sum()
    }

    fn hash_value(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Clone, Default)]
pub struct SystemMemory {
    pub total_memory: u64,
    pub free_memory: u64,
    pub free_swap: u64,
}

#[derive(Clone, Default)]
pub struct SystemInfo {
    pub system: SystemMemory,
}

#[derive(Clone, Default, Debug)]
pub struct RunnerOptions {
    pub num_gpu: i32,
}

#[derive(Clone, Default, Debug)]
pub struct Options {
    pub runner: RunnerOptions,
    pub num_ctx: i32,
    pub num_predict: i32,
}

pub struct LlmServer {
    pub total_layers: u64,
    pub options: Options,
    pub sem: Arc<Semaphore>,
}

impl Default for LlmServer {
    fn default() -> Self {
        Self { total_layers: 0, options: Options::default(), sem: Arc::new(Semaphore::new(1)) }
    }
}

pub struct OllamaServer {
    pub llm: LlmServer,
    pub mem: Option<BackendMemory>,
}

impl Default for OllamaServer {
    fn default() -> Self {
        Self { llm: LlmServer::default(), mem: None }
    }
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum LlmError {
    #[error("invalid format: {0}; expected \"json\" or a valid JSON Schema object")]
    InvalidFormat(String),
    #[error("invalid JSON schema in format")]
    InvalidJsonSchema,
    #[error("load requires full offload")]
    LoadRequiredFull,
    #[error("cancelled")]
    Cancelled,
    #[error("{0}")]
    Other(String),
}

pub struct CompletionRequest {
    pub format: Option<Vec<u8>>,
    pub options: Option<Options>,
}

impl OllamaServer {
    pub fn create_layout(
        &self,
        system_info: &SystemInfo,
        system_gpus: &GpuInfoList,
        memory: Option<&BackendMemory>,
        require_full: bool,
        backoff: f32,
    ) -> Result<GpuLayersList, LlmError> {
        if self.llm.total_layers == 0
            || self.llm.options.runner.num_gpu == 0
            || system_gpus.is_empty()
            || (system_gpus.len() == 1 && system_gpus[0].library == "cpu")
        {
            return Ok(Vec::new());
        }
        let mut gpus = system_gpus.clone();
        gpus.sort_by(|a, b| b.mem_info.free_memory.cmp(&a.mem_info.free_memory));

        let mut mem = memory.cloned().unwrap_or_else(|| BackendMemory {
            cpu: DeviceMemory {
                weights: vec![Memory::default(); self.llm.total_layers as usize],
                cache: vec![Memory::default(); self.llm.total_layers as usize],
                ..Default::default()
            },
            ..Default::default()
        });

        let mut layers = vec![0u64; mem.cpu.weights.len()];
        for i in 0..layers.len() {
            for g in &mem.gpus {
                layers[i] += g.weights[i].size + g.cache[i].size;
            }
            layers[i] += mem.cpu.weights[i].size + mem.cpu.cache[i].size;
        }

        let mut gpu_layers: GpuLayersList = Vec::new();
        for mut gl in gpus.by_library() {
            let mut last_used_gpu = 0usize;
            for i in 0..gl.len() {
                let mut found = false;
                for mg in &mut mem.gpus {
                    if gl[i].id == mg.id {
                        if mg.graph.size != 0 {
                            last_used_gpu = i;
                        }
                        let reserved = (gl[i].mem_info.free_memory as f32 * backoff) as u64
                            + gl[i].minimum_memory
                            + gpu_overhead()
                            + mg.graph.size;
                        if gl[i].mem_info.free_memory > reserved {
                            gl[i].mem_info.free_memory -= reserved;
                        } else {
                            gl[i].mem_info.free_memory = 0;
                        }
                        found = true;
                        break;
                    }
                }
                if !found {
                    gl[i].mem_info.free_memory = 0;
                }
            }
            let library_gpu_layers = assign_layers(
                &layers,
                &gl,
                self.llm.options.runner.num_gpu,
                last_used_gpu,
            );
            if library_gpu_layers.sum() > gpu_layers.sum() {
                gpu_layers = library_gpu_layers;
            }
        }

        let mut cpu_size = mem.input_weights.size + mem.cpu.graph.size;
        let mut vram_size = 0u64;
        for gl in &gpu_layers {
            if let Some(g) = mem.gpus.iter().find(|g| g.id == gl.id) {
                vram_size += g.graph.size;
            }
        }
        for (i, layer_size) in layers.iter().enumerate() {
            if gpu_layers.iter().any(|g| g.layers.contains(&i)) {
                vram_size += layer_size;
            } else {
                cpu_size += layer_size;
            }
        }
        if require_full {
            if gpu_layers.sum() < layers.len()
                && (self.llm.options.runner.num_gpu < 0
                    || gpu_layers.sum() < self.llm.options.runner.num_gpu as usize)
            {
                return Err(LlmError::LoadRequiredFull);
            }
            if cpu_size > system_info.system.free_memory {
                return Err(LlmError::LoadRequiredFull);
            }
        }
        let available = system_info.system.free_memory + system_info.system.free_swap;
        if cpu_size > available {
            return Err(LlmError::Other(format!(
                "model requires more system memory ({} bytes) than is available ({} bytes)",
                cpu_size, available
            )));
        }
        if gpu_layers.sum() == 0 {
            // log::debug!("insufficient VRAM to load any model layers");
        }
        Ok(gpu_layers)
    }

    pub async fn completion(
        &self,
        ctx: &CancellationToken,
        mut req: CompletionRequest,
    ) -> Result<(), LlmError> {
        if let Some(format) = &req.format {
            if !format.is_empty() {
                let s = String::from_utf8_lossy(format);
                match s.as_ref() {
                    "null" | "\"\"" => {}
                    "\"json\"" => {}
                    _ => {
                        if !s.trim_start().starts_with('{') {
                            return Err(LlmError::InvalidFormat(s.to_string()));
                        }
                        if serde_json::from_slice::<serde_json::Value>(format).is_err() {
                            return Err(LlmError::InvalidJsonSchema);
                        }
                    }
                }
            }
        }
        if req.options.is_none() {
            req.options = Some(Options::default());
        }
        if ctx.is_cancelled() {
            return Err(LlmError::Cancelled);
        }
        tokio::select! {
            _ = ctx.cancelled() => Err(LlmError::Cancelled),
            permit = self.llm.sem.acquire() => {
                drop(permit);
                Ok(())
            }
        }
    }
}

fn gpu_overhead() -> u64 {
    0
}

fn sched_spread() -> bool {
    false
}

fn assign_layers(
    initial_layers: &[u64],
    gpus: &GpuInfoList,
    requested_layers: i32,
    last_used_gpu: usize,
) -> GpuLayersList {
    let mut layers = initial_layers.to_vec();
    let mut result = Vec::new();
    for _ in 0..2 {
        let req = if requested_layers < 0 {
            requested_layers
        } else {
            std::cmp::min(layers.len() as i32, requested_layers)
        };
        if !sched_spread() {
            for i in last_used_gpu..gpus.len() {
                let force_request = i == gpus.len() - 1;
                result = find_best_fit(&layers, &gpus[..=i].to_vec(), req, force_request);
                if result.sum() == layers.len() || result.sum() == req as usize {
                    break;
                }
            }
        } else {
            result = find_best_fit(&layers, gpus, req, true);
        }
        if result.sum() == layers.len() {
            return result;
        }
        if layers.is_empty() {
            break;
        }
        layers.pop();
    }
    result
}

fn find_best_fit(
    layers: &[u64],
    gpus: &GpuInfoList,
    requested_layers: i32,
    force_request: bool,
) -> GpuLayersList {
    let mut high: f32 = if requested_layers >= 0 && force_request { 1000.0 } else { 1.0 };
    let mut low: f32 = 0.0;
    let mut best = greedy_fit(layers, gpus, high, requested_layers);
    let max_num_gpu = best.sum();
    if max_num_gpu == 0 {
        return best;
    }
    while high - low > 1e-6 {
        let mid = (low + high) / 2.0;
        let assignments = greedy_fit(layers, gpus, mid, requested_layers);
        if assignments.sum() == max_num_gpu {
            high = mid;
            best = assignments;
        } else {
            low = mid;
        }
    }
    best
}

fn greedy_fit(
    layers: &[u64],
    gpus: &GpuInfoList,
    capacity: f32,
    requested_layers: i32,
) -> GpuLayersList {
    if gpus.is_empty() {
        return Vec::new();
    }
    let mut device: isize = gpus.len() as isize - 1;
    let mut list = vec![GpuLayers { id: gpus[device as usize].id.clone(), layers: Vec::new() }];
    let mut free_space = (gpus[device as usize].mem_info.free_memory as f32 * capacity) as u64;
    for i in (0..layers.len()).rev() {
        if requested_layers >= 0 && (layers.len() - 1 - i) >= requested_layers as usize {
            break;
        }
        loop {
            if layers[i] <= free_space {
                list[0].layers.insert(0, i);
                free_space -= layers[i];
                break;
            }
            device -= 1;
            if device < 0 {
                list.retain(|g| !g.layers.is_empty());
                return list;
            }
            list.insert(0, GpuLayers { id: gpus[device as usize].id.clone(), layers: Vec::new() });
            free_space = (gpus[device as usize].mem_info.free_memory as f32 * capacity) as u64;
        }
    }
    list.retain(|g| !g.layers.is_empty());
    list
}
