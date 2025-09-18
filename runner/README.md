# `runner`

The legacy Go runner has been replaced by the Rust crate in `rust/runner`.
Use the crate as a library or integrate it via the workspace binaries:

```shell
cargo test -p runner
```

The crate exposes shared traits and helpers that back the CLI and server components.
