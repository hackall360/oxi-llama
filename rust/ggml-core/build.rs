fn main() {
    // Always build the reference quantization helpers so tests can link
    cc::Build::new()
        .file("tests/quant_ref.c")
        .compile("quant_ref");
}
