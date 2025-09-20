# Development

Install prerequisites:

- [Rust toolchain](https://rustup.rs/) (see [Rust setup](#rust-setup) for detailed steps)
- C/C++ Compiler e.g. Clang on macOS, [TDM-GCC](https://github.com/jmeubank/tdm-gcc/releases/latest) (Windows amd64) or [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) (Windows arm64), GCC/Clang on Linux.

Then build and run Ollama from the root directory of the repository:

```shell
cargo run -- serve
```

## Rust setup

Ollama's server and command-line interface are implemented entirely in Rust. The
repository is a Cargo workspace containing the binary target in `src/` and the
supporting crates under `rust/`.

1. Install `rustup` from [rustup.rs](https://rustup.rs) and select the `stable`
   toolchain (the project tests against the latest stable release).
2. Install the components used by the CI checks:

   ```shell
   rustup component add rustfmt clippy rust-analyzer
   ```

3. (Optional) Install additional targets when cross-compiling:

   ```shell
   rustup target add aarch64-apple-darwin x86_64-apple-darwin \
     aarch64-pc-windows-msvc x86_64-pc-windows-msvc
   ```

4. Confirm the toolchain is available:

   ```shell
   cargo --version
   ```

### Common Cargo commands

- `cargo run -- serve` – start the Ollama daemon directly from the workspace.
- `cargo build --release` – compile optimized binaries for packaging.
- `cargo test --all` – execute the workspace test suite.
- `cargo fmt --all --check` – ensure formatting matches `rustfmt`.
- `cargo clippy --all-targets --all-features -- -D warnings` – lint the code.

### GPU features

GPU acceleration is controlled through Cargo features exposed by the `ml`
crate:

- Enable the Torch runtime with `ml/tch`.
- Select a backend with `tch/cuda` (NVIDIA) or `tch/rocm` (AMD).

You can combine these features when building locally:

```shell
cargo build --release --features ml/tch,tch/cuda
```

When building in Docker, pass `CARGO_FEATURES=ml/tch,tch/<backend>` as shown in
the [Docker section](#docker).

## macOS (Apple Silicon)

macOS Apple Silicon supports Metal which is built-in to the Ollama binary. No additional steps are required.

## macOS (Intel)

Install prerequisites:

- [CMake](https://cmake.org/download/) or `brew install cmake`

Then, configure and build the project:

```shell
cmake -B build
cmake --build build
```

Lastly, run Ollama:

```shell
cargo run -- serve
```

## Windows

Install prerequisites:

- [CMake](https://cmake.org/download/)
- [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) including the Native Desktop Workload
- (Optional) AMD GPU support
    - [ROCm](https://rocm.docs.amd.com/en/latest/)
    - [Ninja](https://github.com/ninja-build/ninja/releases)
- (Optional) NVIDIA GPU support
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network)

Then, configure and build the project:

```shell
cmake -B build
cmake --build build --config Release
```

> [!IMPORTANT]
> Building for ROCm requires additional flags:
> ```
> cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
> cmake --build build --config Release
> ```


Lastly, run Ollama:

```shell
cargo run -- serve
```

## Windows (ARM)

Windows ARM does not support additional acceleration libraries at this time.  Do not use cmake, simply `cargo run -- serve`.

## Linux

Install prerequisites:

- [CMake](https://cmake.org/download/) or `sudo apt install cmake` or `sudo dnf install cmake`
- (Optional) AMD GPU support
    - [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)
- (Optional) NVIDIA GPU support
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads)

> [!IMPORTANT]
> Ensure prerequisites are in `PATH` before running CMake.


Then, configure and build the project:

```shell
cmake -B build
cmake --build build
```

Lastly, run Ollama:

```shell
cargo run -- serve
```

## Docker

```shell
docker build .
```

Optional build arguments are available to tune the Rust compilation that runs in
the Docker builder stage:

- `CARGO_FEATURES` – comma-separated list passed to `cargo build --features`.
  Enable `ml/tch` for the Torch-based runtime and add `tch/cuda` or
  `tch/rocm` to match the GPU backend that is being packaged.
- `RUST_TARGETS` – space-separated `rustup target add` values that are
  installed before building (useful for cross-compilation).

Examples:

```shell
docker build --build-arg CARGO_FEATURES=ml/tch .
docker build --build-arg FLAVOR=rocm --build-arg CARGO_FEATURES=ml/tch,tch/rocm .
```

### ROCm

```shell
docker build --build-arg FLAVOR=rocm .
```

## Running tests

To run tests, use Cargo:

```shell
cargo test --all
```

Format and lint the workspace before submitting changes:

```shell
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
```

## Library detection

Ollama looks for acceleration libraries in the following paths relative to the `ollama` executable:

* `./lib/ollama` (Windows)
* `../lib/ollama` (Linux)
* `.` (macOS)
* `build/lib/ollama` (for development)

If the libraries are not found, Ollama will not run with any acceleration libraries.
