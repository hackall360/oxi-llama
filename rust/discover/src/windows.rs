use crate::{GpuInfo, MemInfo};
use std::mem::size_of;
use sysinfo::SystemExt;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GroupAffinity {
    pub mask: usize,
    pub group: u16,
    pub reserved: [u16; 3],
}

impl GroupAffinity {
    fn is_member(&self, target: &GroupAffinity) -> bool {
        self.mask & target.mask != 0
    }
}

#[derive(Default, Clone)]
pub struct WinPackage {
    pub groups: Vec<GroupAffinity>,
    pub core_count: i32,
    pub efficiency_core_count: i32,
    pub thread_count: i32,
}

impl WinPackage {
    fn is_member(&self, target: &GroupAffinity) -> bool {
        self.groups.iter().any(|g| g.is_member(target))
    }
}

#[repr(C)]
struct SystemLogicalProcessorInformationExHeader {
    relationship: u32,
    size: u32,
}

#[repr(C)]
struct ProcessorRelationship {
    flags: u8,
    efficiency_class: u8,
    reserved: [u8; 20],
    group_count: u16,
    group_mask: [GroupAffinity; 1],
}

const RELATION_PROCESSOR_CORE: u32 = 0;
const RELATION_PROCESSOR_PACKAGE: u32 = 3;

pub fn process_system_logical_processor_information_list(buf: &[u8]) -> Vec<WinPackage> {
    let mut packages: Vec<WinPackage> = Vec::new();
    let mut offset: usize = 0;
    // first pass: collect packages
    while offset < buf.len() {
        unsafe {
            let header =
                &*(buf[offset..].as_ptr() as *const SystemLogicalProcessorInformationExHeader);
            if header.relationship == RELATION_PROCESSOR_PACKAGE {
                let pr_ptr = buf[offset + size_of::<SystemLogicalProcessorInformationExHeader>()..]
                    .as_ptr() as *const ProcessorRelationship;
                let pr = &*pr_ptr;
                let mut pkg = WinPackage::default();
                let mut ga_ptr = pr.group_mask.as_ptr();
                for _ in 0..pr.group_count {
                    pkg.groups.push(*ga_ptr);
                    ga_ptr = ga_ptr.add(1);
                }
                packages.push(pkg);
            }
            offset += header.size as usize;
        }
    }

    // second pass: determine max efficiency class
    let mut max_eff = 0u8;
    offset = 0;
    while offset < buf.len() {
        unsafe {
            let header =
                &*(buf[offset..].as_ptr() as *const SystemLogicalProcessorInformationExHeader);
            if header.relationship == RELATION_PROCESSOR_CORE {
                let pr_ptr = buf[offset + size_of::<SystemLogicalProcessorInformationExHeader>()..]
                    .as_ptr() as *const ProcessorRelationship;
                let pr = &*pr_ptr;
                if pr.efficiency_class > max_eff {
                    max_eff = pr.efficiency_class;
                }
            }
            offset += header.size as usize;
        }
    }

    // third pass: count cores, threads, and efficiency cores
    offset = 0;
    while offset < buf.len() {
        unsafe {
            let header =
                &*(buf[offset..].as_ptr() as *const SystemLogicalProcessorInformationExHeader);
            if header.relationship == RELATION_PROCESSOR_CORE {
                let pr_ptr = buf[offset + size_of::<SystemLogicalProcessorInformationExHeader>()..]
                    .as_ptr() as *const ProcessorRelationship;
                let pr = &*pr_ptr;
                let mut ga_ptr = pr.group_mask.as_ptr();
                for _ in 0..pr.group_count {
                    let gm = &*ga_ptr;
                    for pkg in packages.iter_mut() {
                        if pkg.is_member(gm) {
                            pkg.core_count += 1;
                            if pr.flags == 0 {
                                pkg.thread_count += 1;
                            } else {
                                pkg.thread_count += 2;
                            }
                            if pr.efficiency_class < max_eff {
                                pkg.efficiency_core_count += 1;
                            }
                        }
                    }
                    ga_ptr = ga_ptr.add(1);
                }
            }
            offset += header.size as usize;
        }
    }

    packages
}

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
