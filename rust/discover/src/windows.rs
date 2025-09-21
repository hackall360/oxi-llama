use crate::{path, GpuInfo, MemInfo};
use nvml_wrapper::{
    bitmasks::InitFlags, cuda_driver_version_major, cuda_driver_version_minor, error::NvmlError,
    Nvml,
};
use std::mem::size_of;
use sysinfo::SystemExt;

mod hip;

#[allow(
    dead_code,
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    unused_imports
)]
mod oneapi_bindings {
    include!(concat!(env!("OUT_DIR"), "/oneapi_bindings.rs"));
}

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

pub struct GpuLoaders<'a> {
    cuda: &'a dyn Fn() -> Result<Vec<GpuInfo>, String>,
    hip: &'a dyn Fn() -> Result<Vec<GpuInfo>, String>,
    oneapi: &'a dyn Fn() -> Result<Vec<GpuInfo>, String>,
}

impl<'a> GpuLoaders<'a> {
    pub fn new(
        cuda: &'a dyn Fn() -> Result<Vec<GpuInfo>, String>,
        hip: &'a dyn Fn() -> Result<Vec<GpuInfo>, String>,
        oneapi: &'a dyn Fn() -> Result<Vec<GpuInfo>, String>,
    ) -> Self {
        Self { cuda, hip, oneapi }
    }
}

fn default_loaders() -> GpuLoaders<'static> {
    GpuLoaders::new(&load_cuda, &hip::collect_hip_info, &oneapi::collect_oneapi_info)
}

pub fn get_gpu_info() -> Vec<GpuInfo> {
    get_gpu_info_with(&default_loaders())
}

pub fn get_gpu_info_with(loaders: &GpuLoaders) -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    if let Ok(mut cuda_gpus) = (loaders.cuda)() {
        gpus.append(&mut cuda_gpus);
    }

    if let Ok(mut hip_gpus) = (loaders.hip)() {
        gpus.append(&mut hip_gpus);
    }

    if let Ok(mut oneapi_gpus) = (loaders.oneapi)() {
        gpus.append(&mut oneapi_gpus);
    }

    if gpus.is_empty() {
        cpu_fallback()
    } else {
        gpus
    }
}

fn load_cuda() -> Result<Vec<GpuInfo>, String> {
    let nvml = Nvml::init_with_flags(InitFlags::NO_GPUS | InitFlags::NO_ATTACH)
        .map_err(|err| err.to_string())?;
    let result = collect_cuda_info(&nvml).map_err(|err| err.to_string());
    let _ = nvml.shutdown();
    result
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
        let device = match nvml.device_by_index(index) {
            Ok(device) => device,
            Err(_) => continue,
        };

        let memory = match device.memory_info() {
            Ok(memory) => memory,
            Err(_) => continue,
        };

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

mod oneapi {
    use super::oneapi_bindings as ze;
    use crate::{path, GpuInfo, MemInfo};
    use libloading::Library;
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_void};
    use std::ptr;
    use std::str::from_utf8;

    const CANDIDATE_LIBRARIES: &[&str] = &["ze_loader.dll", "ze_loader64.dll", "libze_loader.dll"];

    type ZesInit = unsafe extern "C" fn(ze::ze_init_flags_t) -> ze::ze_result_t;
    type ZesDriverGet =
        unsafe extern "C" fn(*mut u32, *mut ze::zes_driver_handle_t) -> ze::ze_result_t;
    type ZesDeviceGet = unsafe extern "C" fn(
        ze::zes_driver_handle_t,
        *mut u32,
        *mut ze::zes_device_handle_t,
    ) -> ze::ze_result_t;
    type ZesDeviceGetProperties = unsafe extern "C" fn(
        ze::zes_device_handle_t,
        *mut ze::zes_device_properties_t,
    ) -> ze::ze_result_t;
    type ZesDeviceEnumMemoryModules = unsafe extern "C" fn(
        ze::zes_device_handle_t,
        *mut u32,
        *mut ze::zes_mem_handle_t,
    ) -> ze::ze_result_t;
    type ZesMemoryGetState =
        unsafe extern "C" fn(ze::zes_mem_handle_t, *mut ze::zes_mem_state_t) -> ze::ze_result_t;
    type ZeDriverGetProperties = unsafe extern "C" fn(
        ze::ze_driver_handle_t,
        *mut ze::ze_driver_properties_t,
    ) -> ze::ze_result_t;

    pub(super) fn collect_oneapi_info() -> Result<Vec<GpuInfo>, String> {
        let library = OneApiLibrary::load()?;
        unsafe { library.enumerate_devices() }
    }

    struct OneApiLibrary {
        _lib: Library,
        zes_init: ZesInit,
        zes_driver_get: ZesDriverGet,
        zes_device_get: ZesDeviceGet,
        zes_device_get_properties: ZesDeviceGetProperties,
        zes_device_enum_memory_modules: ZesDeviceEnumMemoryModules,
        zes_memory_get_state: ZesMemoryGetState,
        ze_driver_get_properties: ZeDriverGetProperties,
    }

    impl OneApiLibrary {
        fn load() -> Result<Self, String> {
            let mut errors = Vec::new();

            for candidate in CANDIDATE_LIBRARIES {
                match unsafe { Library::new(candidate) } {
                    Ok(lib) => match unsafe { Self::from_library(lib) } {
                        Ok(loaded) => return Ok(loaded),
                        Err(err) => errors.push(format!("{}: {}", candidate, err)),
                    },
                    Err(err) => errors.push(format!("{}: {}", candidate, err)),
                }
            }

            if errors.is_empty() {
                Err("failed to load Level Zero loader library".to_string())
            } else {
                Err(format!(
                    "failed to load Level Zero loader library: {}",
                    errors.join("; ")
                ))
            }
        }

        unsafe fn from_library(lib: Library) -> Result<Self, String> {
            Ok(Self {
                zes_init: load_symbol(&lib, b"zesInit\0")?,
                zes_driver_get: load_symbol(&lib, b"zesDriverGet\0")?,
                zes_device_get: load_symbol(&lib, b"zesDeviceGet\0")?,
                zes_device_get_properties: load_symbol(&lib, b"zesDeviceGetProperties\0")?,
                zes_device_enum_memory_modules: load_symbol(&lib, b"zesDeviceEnumMemoryModules\0")?,
                zes_memory_get_state: load_symbol(&lib, b"zesMemoryGetState\0")?,
                ze_driver_get_properties: load_symbol(&lib, b"zeDriverGetProperties\0")?,
                _lib: lib,
            })
        }

        unsafe fn enumerate_devices(&self) -> Result<Vec<GpuInfo>, String> {
            ze_check((self.zes_init)(0), "zesInit")?;

            let mut driver_count: u32 = 0;
            ze_check(
                (self.zes_driver_get)(&mut driver_count as *mut u32, ptr::null_mut()),
                "zesDriverGet",
            )?;

            if driver_count == 0 {
                return Ok(Vec::new());
            }

            let mut drivers = vec![ptr::null_mut(); driver_count as usize];
            ze_check(
                (self.zes_driver_get)(&mut driver_count as *mut u32, drivers.as_mut_ptr()),
                "zesDriverGet",
            )?;

            let mut gpus = Vec::new();

            for (driver_index, driver) in drivers.iter().enumerate() {
                if driver.is_null() {
                    continue;
                }

                let driver_details = self.driver_details(*driver)?;

                let mut device_count: u32 = 0;
                ze_check(
                    (self.zes_device_get)(*driver, &mut device_count as *mut u32, ptr::null_mut()),
                    "zesDeviceGet",
                )?;

                if device_count == 0 {
                    continue;
                }

                let mut devices = vec![ptr::null_mut(); device_count as usize];
                ze_check(
                    (self.zes_device_get)(
                        *driver,
                        &mut device_count as *mut u32,
                        devices.as_mut_ptr(),
                    ),
                    "zesDeviceGet",
                )?;

                for (device_index, device) in devices.iter().enumerate() {
                    if device.is_null() {
                        continue;
                    }

                    let properties = self.device_properties(*device).map_err(|err| {
                        format!("{} (driver {}, device {})", err, driver_index, device_index)
                    })?;

                    let mem_info = self.device_mem_info(*device).map_err(|err| {
                        format!("{} (driver {}, device {})", err, driver_index, device_index)
                    })?;

                    let (mut driver_major, mut driver_minor) = driver_details.as_pair();
                    if let Some((major, minor)) = parse_driver_version(&properties.driver_version) {
                        driver_major = major;
                        driver_minor = minor;
                    }

                    let variant = if driver_major > 0 {
                        format!("v{}", driver_major)
                    } else {
                        String::new()
                    };
                    let dependency_paths = path::default_dependency_paths("oneapi", &variant);

                    let id = properties
                        .uuid
                        .unwrap_or_else(|| format!("oneapi-{}-{}", driver_index, device_index));

                    gpus.push(GpuInfo {
                        mem_info,
                        library: "oneapi".into(),
                        variant,
                        dependency_path: dependency_paths,
                        id,
                        name: properties.name,
                        compute: String::new(),
                        driver_major,
                        driver_minor,
                        ..Default::default()
                    });
                }
            }

            Ok(gpus)
        }

        unsafe fn driver_details(
            &self,
            driver: ze::zes_driver_handle_t,
        ) -> Result<DriverDetails, String> {
            let mut props: ze::ze_driver_properties_t = std::mem::zeroed();
            props.stype = ze::_ze_structure_type_t_ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
            props.pNext = ptr::null_mut();
            ze_check(
                (self.ze_driver_get_properties)(
                    driver,
                    &mut props as *mut ze::ze_driver_properties_t,
                ),
                "zeDriverGetProperties",
            )?;

            Ok(DriverDetails::new(props.driverVersion))
        }

        unsafe fn device_properties(
            &self,
            device: ze::zes_device_handle_t,
        ) -> Result<DeviceProperties, String> {
            let mut ext_props: ze::zes_device_ext_properties_t = std::mem::zeroed();
            ext_props.stype = ze::_zes_structure_type_t_ZES_STRUCTURE_TYPE_DEVICE_EXT_PROPERTIES;
            ext_props.pNext = ptr::null_mut();

            let mut props: ze::zes_device_properties_t = std::mem::zeroed();
            props.stype = ze::_zes_structure_type_t_ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            props.pNext = &mut ext_props as *mut _ as *mut c_void;

            ze_check(
                (self.zes_device_get_properties)(
                    device,
                    &mut props as *mut ze::zes_device_properties_t,
                ),
                "zesDeviceGetProperties",
            )?;

            let name = c_char_array_to_string(&props.modelName);
            let driver_version = c_char_array_to_string(&props.driverVersion);
            let uuid = uuid_to_string(&ext_props.uuid.id);

            Ok(DeviceProperties {
                name,
                driver_version,
                uuid,
            })
        }

        unsafe fn device_mem_info(
            &self,
            device: ze::zes_device_handle_t,
        ) -> Result<MemInfo, String> {
            let mut module_count: u32 = 0;
            ze_check(
                (self.zes_device_enum_memory_modules)(
                    device,
                    &mut module_count as *mut u32,
                    ptr::null_mut(),
                ),
                "zesDeviceEnumMemoryModules",
            )?;

            if module_count == 0 {
                return Ok(MemInfo::default());
            }

            let mut modules = vec![ptr::null_mut(); module_count as usize];
            ze_check(
                (self.zes_device_enum_memory_modules)(
                    device,
                    &mut module_count as *mut u32,
                    modules.as_mut_ptr(),
                ),
                "zesDeviceEnumMemoryModules",
            )?;

            let mut total = 0u64;
            let mut free = 0u64;

            for module in modules {
                if module.is_null() {
                    continue;
                }

                let mut state: ze::zes_mem_state_t = std::mem::zeroed();
                state.stype = ze::_zes_structure_type_t_ZES_STRUCTURE_TYPE_MEM_STATE;
                state.pNext = ptr::null();

                ze_check(
                    (self.zes_memory_get_state)(module, &mut state as *mut ze::zes_mem_state_t),
                    "zesMemoryGetState",
                )?;

                total = total.saturating_add(state.size);
                free = free.saturating_add(state.free);
            }

            Ok(MemInfo {
                total_memory: total,
                free_memory: free,
                free_swap: 0,
            })
        }
    }

    #[derive(Default)]
    struct DriverDetails {
        version: u32,
    }

    impl DriverDetails {
        fn new(version: u32) -> Self {
            Self { version }
        }

        fn as_pair(&self) -> (i32, i32) {
            decode_driver_version(self.version)
        }
    }

    struct DeviceProperties {
        name: String,
        driver_version: String,
        uuid: Option<String>,
    }

    unsafe fn load_symbol<T>(lib: &Library, name: &[u8]) -> Result<T, String>
    where
        T: Copy,
    {
        lib.get::<T>(name)
            .map(|symbol| *symbol)
            .map_err(|err| format!("{}: {}", symbol_label(name), err))
    }

    fn ze_check(result: ze::ze_result_t, func: &str) -> Result<(), String> {
        if result == ze::_ze_result_t_ZE_RESULT_SUCCESS {
            Ok(())
        } else {
            Err(format!("{} failed with status {:#x}", func, result as i32))
        }
    }

    fn parse_driver_version(version: &str) -> Option<(i32, i32)> {
        let mut parts = version
            .split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty());

        let major = parts.next()?.parse::<i32>().ok()?;
        let minor = parts
            .next()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);

        Some((major, minor))
    }

    fn decode_driver_version(version: u32) -> (i32, i32) {
        if version == 0 {
            return (0, 0);
        }

        let major = ((version >> 16) & 0xffff) as i32;
        let minor = (version & 0xffff) as i32;

        if major > 0 {
            (major, minor)
        } else {
            let major = (version / 10000) as i32;
            let minor = ((version % 10000) / 100) as i32;
            if major > 0 {
                (major, minor)
            } else {
                (version as i32, 0)
            }
        }
    }

    fn uuid_to_string(bytes: &[u8]) -> Option<String> {
        if bytes.iter().all(|b| *b == 0) {
            return None;
        }

        let mut out = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            out.push_str(&format!("{:02x}", byte));
        }
        Some(out)
    }

    fn c_char_array_to_string(buffer: &[c_char]) -> String {
        if buffer.is_empty() {
            return String::new();
        }

        unsafe {
            CStr::from_ptr(buffer.as_ptr())
                .to_string_lossy()
                .trim_end_matches('\0')
                .trim()
                .to_string()
        }
    }

    fn symbol_label(name: &[u8]) -> &str {
        let end = name.iter().position(|b| *b == 0).unwrap_or(name.len());
        from_utf8(&name[..end]).unwrap_or("<invalid>")
    }
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
