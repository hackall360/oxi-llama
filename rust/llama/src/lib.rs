use std::ffi::CString;

// Bindings to the C helper are generated at build time by bindgen and included
// here. This keeps the Rust declarations in sync with the C header.
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Convert a JSON schema into a grammar representation used by llama.cpp.
///
/// Returns `None` if the provided schema is invalid.
pub fn schema_to_grammar_safe(schema: &str) -> Option<Vec<u8>> {
    let c_schema = CString::new(schema).ok()?;
    // similar heuristic to llama.go
    let max_len = std::cmp::max(32 * 1024, std::cmp::min(1024 * 1024, schema.len() * 4));
    let mut buf = vec![0u8; max_len];
    let n = unsafe { schema_to_grammar(c_schema.as_ptr(), buf.as_mut_ptr() as *mut ::std::os::raw::c_char, max_len) };
    if n <= 0 {
        None
    } else {
        buf.truncate(n as usize);
        Some(buf)
    }
}
