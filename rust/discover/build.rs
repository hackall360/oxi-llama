use std::env;
use std::path::PathBuf;

#[cfg(target_os = "linux")]
fn main() {
    generate_bindings();
}

#[cfg(not(target_os = "linux"))]
fn main() {}

#[cfg(target_os = "linux")]
fn generate_bindings() {
    println!("cargo:rerun-if-changed=oneapi/ze_api.h");
    println!("cargo:rerun-if-changed=oneapi/zes_api.h");

    let header = PathBuf::from("oneapi/zes_api.h");
    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .clang_arg("-Ioneapi")
        .allowlist_type("ze_.*")
        .allowlist_type("zes_.*")
        .allowlist_var("ZE_.*")
        .allowlist_var("ZES_.*")
        .allowlist_function("zeDriverGetProperties")
        .allowlist_function("zesInit")
        .allowlist_function("zesDriverGet")
        .allowlist_function("zesDeviceGet")
        .allowlist_function("zesDeviceGetProperties")
        .allowlist_function("zesDeviceEnumMemoryModules")
        .allowlist_function("zesMemoryGetState")
        .generate()
        .expect("unable to generate Level Zero bindings");

    bindings
        .write_to_file(out_path.join("oneapi_bindings.rs"))
        .expect("could not write Level Zero bindings");
}
