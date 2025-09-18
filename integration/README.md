# Integration Tests

This directory now hosts shared test data used by the Rust integration test suite
located under `rust/tests`.

Run the integration tests from the repository root with:

```shell
cargo test -p oxi_rust
```

By default the tests will launch the Ollama server, exercise the endpoints, and shut it down.
Set `OLLAMA_TEST_EXISTING` to target an already-running instance.
