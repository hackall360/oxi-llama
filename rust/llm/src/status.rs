use std::fs::File;
use std::io::{self, Write};

/// Captures error messages from the runner process.
pub struct StatusWriter {
    pub last_err_msg: String,
    out: File,
}

impl StatusWriter {
    pub fn new(out: File) -> Self {
        Self { last_err_msg: String::new(), out }
    }
}

// Known prefixes that indicate an error in the subprocess output.
const ERROR_PREFIXES: [&str; 7] = [
    "error:",
    "CUDA error",
    "cudaMalloc failed",
    "\"ERR\"",
    "error loading model",
    "GGML_ASSERT",
    "Deepseek2 does not support K-shift",
];

impl Write for StatusWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        for prefix in ERROR_PREFIXES.iter() {
            if let Some(pos) = buf.windows(prefix.len()).position(|w| w == prefix.as_bytes()) {
                let after = &buf[pos + prefix.len()..];
                let msg = format!("{}{}", prefix, String::from_utf8_lossy(after).trim());
                self.last_err_msg = msg;
            }
        }
        self.out.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}
