# Rust Migration Inventory

The Go implementation of the Ollama daemon has been fully retired. All runtime
services, the CLI, and supporting libraries now live in the Rust workspace
rooted at `Cargo.toml` and `rust/`. This page captures the historical
translation of key components and the limited set of projects that intentionally
remain in other languages.

## Completed migrations

| Component | Previous Implementation | Replacement |
|-----------|-------------------------|-------------|
| Core server, runner, model, and CLI layers | Go modules spread across `./server`, `./runner`, `./model`, and friends | Workspace crates under `rust/server`, `rust/runner`, `rust/model`, `rust/cli`, and the binary entry point in `src/main.rs` |
| Filesystem utilities such as `fs/util/bufioutil` | Go helper wrapping `bufio.Reader` for seekable readers | Rust translation in `rust/fs/src/util/bufioutil.rs` exposed via `fs::util::bufioutil::BufferedSeeker` |
| Prompt templating | Go package embedding `template/` assets | Rust `template` crate embedding the assets and providing `Template`, `Values`, and named helpers |

Each workspace crate can be built and tested individually, but all targets are
also exercised from the repository root:

```bash
cargo test --all
```

## Intentional non-Rust components

The following projects continue to use platform-native tooling:

| Component | Language / Tooling | Rationale |
|-----------|--------------------|-----------|
| `macapp/` | Electron/TypeScript | Provides the cross-platform desktop UI that consumes the Rust HTTP/gRPC APIs. |
| `installer/setup.iss` | Inno Setup script | Windows packaging and shortcuts are authored in Inno Setup and invoked after the Rust binaries are produced. |
| `scripts/` and `installer/` shell helpers | POSIX shell & PowerShell | Retained for packaging, release automation, and integration with OS-specific installers. |

These components interface with the Rust binaries but do not contain runtime
business logic.
