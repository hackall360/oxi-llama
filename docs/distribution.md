# Linux Distribution Bundles

The Rust-based release pipeline emits architecture-specific tarballs that share a common
layout. All archives are compressed as `tar.gz` files and extract directly into the target
prefix (typically `/usr`).

## Artifact naming

| Artifact | Purpose | Contents |
| --- | --- | --- |
| `ollama-linux-amd64.tar.gz` | Primary runtime for x86_64 hosts | Installs the Rust `ollama` CLI into `/usr/bin` and CPU/CUDA runtime libraries into `/usr/lib/ollama`. |
| `ollama-linux-arm64.tar.gz` | Primary runtime for ARM64 hosts | Same layout as the amd64 bundle, built for `aarch64`. |
| `ollama-linux-amd64-gpu-rocm.tar.gz` | Optional AMD GPU support for x86_64 | Adds ROCm libraries under `/usr/lib/ollama/rocm`; unpack alongside the base bundle. |
| `ollama-linux-arm64-gpu-jetpack5.tar.gz` | JetPack 5 CUDA runtime | Provides `/usr/lib/ollama/cuda_jetpack5` for NVIDIA Jetson devices running JetPack 5. |
| `ollama-linux-arm64-gpu-jetpack6.tar.gz` | JetPack 6 CUDA runtime | Provides `/usr/lib/ollama/cuda_jetpack6` for JetPack 6 systems. |

> ℹ️ Legacy `.tgz` filenames remain available for older releases. The installer will fall back
to those names when fetching historic versions via `OLLAMA_VERSION`.

## Directory structure

Each archive maintains the following layout relative to the installation prefix:

```
/usr/
  bin/
    ollama
  lib/
    ollama/
      <runtime libraries>
```

The base bundle includes the Rust binary and supporting shared libraries in `lib/ollama`.
GPU-specific bundles only contain their respective subdirectories (for example,
`lib/ollama/rocm` or `lib/ollama/cuda_jetpack6`).

## Testing downloads locally

Set `OLLAMA_DOWNLOAD_BASE_URL` to a custom `file://` or HTTP endpoint to make the
installer consume alternate artifacts:

```bash
OLLAMA_DOWNLOAD_BASE_URL="file:///tmp/ollama-dist" ./scripts/install.sh
```

This is useful when smoke-testing new bundles without publishing them to the public CDN.
