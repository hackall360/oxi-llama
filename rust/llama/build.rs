fn main() {
    // Compile the small C++ helper used to transform JSON schema into a
    // llama.cpp grammar.
    let mut build = cc::Build::new();
    build.cpp(true);
    build.file("../ffi/schema_to_grammar.cpp");
    build.file("../ffi/json-schema-to-grammar.cpp");
    build.include("../ffi");
    build.include("../../llama/llama.cpp/common");
    build.include("../../llama/llama.cpp/vendor");
    build.flag_if_supported("-std=c++17");
    build.compile("schema_to_grammar");

    // Instruct Cargo to rerun this build script when any of the inputs change.
    println!("cargo:rerun-if-changed=../ffi/schema_to_grammar.cpp");
    println!("cargo:rerun-if-changed=../ffi/json-schema-to-grammar.cpp");
    println!("cargo:rerun-if-changed=../ffi/schema_to_grammar.h");
    println!("cargo:rerun-if-changed=../../llama/llama.cpp/common/json-schema-to-grammar.h");

    // Generate Rust bindings for the C header using bindgen so that the FFI
    // surface stays in sync with the C implementation.
    let bindings = bindgen::Builder::default()
        .header("../ffi/schema_to_grammar.h")
        .clang_arg("-I../ffi")
        .clang_arg("-I../../llama/llama.cpp/common")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
