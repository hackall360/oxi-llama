fn main() {
    let mut build = cc::Build::new();
    build.cpp(true);
    build.file("../ffi/schema_to_grammar.cpp");
    build.file("../ffi/json-schema-to-grammar.cpp");
    build.include("../ffi");
    build.include("../../llama/llama.cpp/common");
    build.include("../../llama/llama.cpp/vendor");
    build.flag_if_supported("-std=c++17");
    build.compile("schema_to_grammar");
    println!("cargo:rerun-if-changed=../ffi/schema_to_grammar.cpp");
    println!("cargo:rerun-if-changed=../ffi/json-schema-to-grammar.cpp");
    println!("cargo:rerun-if-changed=../ffi/schema_to_grammar.h");
    println!("cargo:rerun-if-changed=../../llama/llama.cpp/common/json-schema-to-grammar.h");
}
