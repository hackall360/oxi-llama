fn main() {
    cc::Build::new().file("ffi/hello.c").compile("hello");
}
