use crate::{path, GpuInfo, MemInfo};
use libloading::Library;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_uint};
use std::str::from_utf8;

const HIP_SUCCESS: hipError_t = 0;

#[allow(non_camel_case_types)]
type hipError_t = c_int;
#[allow(non_camel_case_types)]
type hipDevice_t = c_int;

type HipInit = unsafe extern "C" fn(c_uint) -> hipError_t;
type HipGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> hipError_t;
type HipDeviceGet = unsafe extern "C" fn(*mut hipDevice_t, c_int) -> hipError_t;
type HipDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, hipDevice_t) -> hipError_t;
type HipDeviceTotalMem = unsafe extern "C" fn(*mut usize, hipDevice_t) -> hipError_t;
type HipMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> hipError_t;
type HipSetDevice = unsafe extern "C" fn(hipDevice_t) -> hipError_t;
type HipDeviceComputeCapability =
    unsafe extern "C" fn(*mut c_int, *mut c_int, hipDevice_t) -> hipError_t;
type HipRuntimeGetVersion = unsafe extern "C" fn(*mut c_int) -> hipError_t;
type HipDriverGetVersion = unsafe extern "C" fn(*mut c_int) -> hipError_t;
type HipDeviceGetPCIBusId = unsafe extern "C" fn(*mut c_char, c_int, hipDevice_t) -> hipError_t;

pub(super) fn collect_hip_info() -> Result<Vec<GpuInfo>, String> {
    let library = match HipLibrary::load() {
        Ok(lib) => lib,
        Err(err) => return Err(err),
    };

    unsafe { library.enumerate_devices() }
}

struct HipLibrary {
    _lib: Library,
    hip_init: HipInit,
    hip_get_device_count: HipGetDeviceCount,
    hip_device_get: HipDeviceGet,
    hip_device_get_name: HipDeviceGetName,
    hip_device_total_mem: HipDeviceTotalMem,
    hip_mem_get_info: HipMemGetInfo,
    hip_set_device: HipSetDevice,
    hip_device_compute_capability: HipDeviceComputeCapability,
    hip_runtime_get_version: Option<HipRuntimeGetVersion>,
    hip_driver_get_version: Option<HipDriverGetVersion>,
    hip_device_get_pci_bus_id: Option<HipDeviceGetPCIBusId>,
}

impl HipLibrary {
    fn load() -> Result<Self, String> {
        let candidates = [
            "libamdhip64.so",
            "libamdhip64.so.6",
            "libamdhip64.so.5",
            "libhip_hcc.so",
        ];

        let mut last_error: Option<String> = None;

        for candidate in candidates.iter() {
            let lib = match unsafe { Library::new(candidate) } {
                Ok(lib) => lib,
                Err(err) => {
                    last_error = Some(format!("{}: {}", candidate, err));
                    continue;
                }
            };

            match unsafe { Self::from_library(lib) } {
                Ok(loaded) => return Ok(loaded),
                Err(err) => last_error = Some(err),
            }
        }

        Err(last_error.unwrap_or_else(|| "failed to load HIP runtime library".to_string()))
    }

    unsafe fn from_library(lib: Library) -> Result<Self, String> {
        Ok(Self {
            hip_init: load_symbol(&lib, b"hipInit\0")?,
            hip_get_device_count: load_symbol(&lib, b"hipGetDeviceCount\0")?,
            hip_device_get: load_symbol(&lib, b"hipDeviceGet\0")?,
            hip_device_get_name: load_symbol(&lib, b"hipDeviceGetName\0")?,
            hip_device_total_mem: load_symbol(&lib, b"hipDeviceTotalMem\0")?,
            hip_mem_get_info: load_symbol(&lib, b"hipMemGetInfo\0")?,
            hip_set_device: load_symbol(&lib, b"hipSetDevice\0")?,
            hip_device_compute_capability: load_symbol(&lib, b"hipDeviceComputeCapability\0")?,
            hip_runtime_get_version: load_optional_symbol(&lib, b"hipRuntimeGetVersion\0"),
            hip_driver_get_version: load_optional_symbol(&lib, b"hipDriverGetVersion\0"),
            hip_device_get_pci_bus_id: load_optional_symbol(&lib, b"hipDeviceGetPCIBusId\0"),
            _lib: lib,
        })
    }

    unsafe fn enumerate_devices(&self) -> Result<Vec<GpuInfo>, String> {
        hip_check((self.hip_init)(0), "hipInit")?;

        let mut count: c_int = 0;
        let status = (self.hip_get_device_count)(&mut count as *mut c_int);
        if status != HIP_SUCCESS {
            return Ok(Vec::new());
        }

        if count <= 0 {
            return Ok(Vec::new());
        }

        let (driver_major, driver_minor) = self.driver_version();
        let variant = self.runtime_variant();
        let dependency_paths = path::default_dependency_paths("rocm", &variant);

        let mut gpus = Vec::with_capacity(count as usize);

        for ordinal in 0..count {
            let mut device: hipDevice_t = 0;
            hip_check(
                (self.hip_device_get)(&mut device as *mut hipDevice_t, ordinal),
                "hipDeviceGet",
            )?;

            let name = self
                .device_name(device)
                .unwrap_or_else(|_| format!("HIP GPU {}", ordinal));
            let id = self
                .device_bus_id(device)
                .unwrap_or_else(|_| format!("hip-{}", ordinal));
            let mem_info = match self.device_mem_info(device) {
                Ok(info) => info,
                Err(_) => continue,
            };
            let compute = self.device_compute_capability(device).unwrap_or_default();

            gpus.push(GpuInfo {
                mem_info,
                library: "rocm".into(),
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

    unsafe fn device_name(&self, device: hipDevice_t) -> Result<String, String> {
        let mut buffer = [0i8; 256];
        hip_check(
            (self.hip_device_get_name)(buffer.as_mut_ptr(), buffer.len() as c_int, device),
            "hipDeviceGetName",
        )?;

        Ok(CStr::from_ptr(buffer.as_ptr())
            .to_string_lossy()
            .trim_end_matches('\0')
            .to_string())
    }

    unsafe fn device_bus_id(&self, device: hipDevice_t) -> Result<String, String> {
        let func = match self.hip_device_get_pci_bus_id {
            Some(f) => f,
            None => return Err("hipDeviceGetPCIBusId symbol missing".to_string()),
        };

        let mut buffer = [0i8; 64];
        hip_check(
            func(buffer.as_mut_ptr(), buffer.len() as c_int, device),
            "hipDeviceGetPCIBusId",
        )?;

        Ok(CStr::from_ptr(buffer.as_ptr())
            .to_string_lossy()
            .trim_end_matches('\0')
            .to_string())
    }

    unsafe fn device_mem_info(&self, device: hipDevice_t) -> Result<MemInfo, String> {
        let mut total_bytes: usize = 0;
        hip_check(
            (self.hip_device_total_mem)(&mut total_bytes as *mut usize, device),
            "hipDeviceTotalMem",
        )?;

        hip_check((self.hip_set_device)(device), "hipSetDevice")?;

        let mut free_bytes: usize = 0;
        let mut _runtime_total: usize = 0;
        hip_check(
            (self.hip_mem_get_info)(
                &mut free_bytes as *mut usize,
                &mut _runtime_total as *mut usize,
            ),
            "hipMemGetInfo",
        )?;

        Ok(MemInfo {
            total_memory: total_bytes as u64,
            free_memory: free_bytes as u64,
            free_swap: 0,
        })
    }

    unsafe fn device_compute_capability(&self, device: hipDevice_t) -> Result<String, String> {
        let mut major: c_int = 0;
        let mut minor: c_int = 0;
        hip_check(
            (self.hip_device_compute_capability)(
                &mut major as *mut c_int,
                &mut minor as *mut c_int,
                device,
            ),
            "hipDeviceComputeCapability",
        )?;

        Ok(format!("{}.{}", major, minor))
    }

    fn driver_version(&self) -> (i32, i32) {
        unsafe {
            if let Some(func) = self.hip_driver_get_version {
                let mut version: c_int = 0;
                if func(&mut version as *mut c_int) == HIP_SUCCESS {
                    return decode_version(version);
                }
            }
        }
        (0, 0)
    }

    fn runtime_variant(&self) -> String {
        unsafe {
            if let Some(func) = self.hip_runtime_get_version {
                let mut version: c_int = 0;
                if func(&mut version as *mut c_int) == HIP_SUCCESS {
                    let (major, _) = decode_version(version);
                    if major > 0 {
                        return format!("v{}", major);
                    }
                }
            }
        }
        String::new()
    }
}

unsafe fn load_symbol<T>(lib: &Library, name: &[u8]) -> Result<T, String>
where
    T: Copy,
{
    lib.get::<T>(name)
        .map(|symbol| *symbol)
        .map_err(|err| format!("{}: {}", symbol_label(name), err))
}

unsafe fn load_optional_symbol<T>(lib: &Library, name: &[u8]) -> Option<T>
where
    T: Copy,
{
    lib.get::<T>(name).map(|symbol| *symbol).ok()
}

fn hip_check(result: hipError_t, func: &str) -> Result<(), String> {
    if result == HIP_SUCCESS {
        Ok(())
    } else {
        Err(format!("{} failed with error {}", func, result))
    }
}

fn decode_version(version: c_int) -> (i32, i32) {
    if version <= 0 {
        return (0, 0);
    }

    let major = version / 1000;
    let minor = (version % 1000) / 10;

    (major, minor)
}

fn symbol_label(name: &[u8]) -> &str {
    let end = name.iter().position(|b| *b == 0).unwrap_or(name.len());
    from_utf8(&name[..end]).unwrap_or("<invalid>")
}
