# Rust Migration Inventory

This document tracks components that previously relied on Go or other tooling and the status of their Rust replacements.

| Component | Previous Implementation | Decision | Replacement |
|-----------|-------------------------|----------|-------------|
| `fs/util/bufioutil` | Go helper wrapping `bufio.Reader` for seekable readers | Rewritten in Rust | `rust/fs/src/util/bufioutil.rs` (exposed via `fs::util::bufioutil::BufferedSeeker`) |
| `template` package | Go prompt templating with embedded assets | Rewritten in Rust | `rust/template` crate embedding `template/` assets and providing `Template`, `Values`, and `named` helpers |
| `installer/setup.iss` | Inno Setup script | Remains platform-native tooling | Windows packaging continues to rely on `installer/setup.iss`; integration with the Rust toolchain is tracked separately |
| `macapp` Electron UI | Node/Electron project | Remains platform-native tooling | Documented as external dependency; Rust back-end exposes APIs consumed by the Electron front-end |

The new Rust crates are part of the `rust/` tree and can be built and tested independently:

```bash
cd rust/fs && cargo test
cd rust/template && cargo test
```

Future work will focus on wiring these crates into the top-level binaries and incrementally retiring the Go code paths once parity is validated in production.
