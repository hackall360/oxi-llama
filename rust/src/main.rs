use oxi_rust::call_c;

fn main() {
    let val = call_c();
    println!("Hello from Rust, C says {val}!");
}
