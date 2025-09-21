# Windows GPU discovery tests

The Windows-specific GPU discovery tests use the `GpuLoaders` abstraction to
mock responses from NVML, HIP, and Level Zero. Because the tests simulate the
vendor libraries, they do **not** need any DLLs on the `PATH` when running in
CI or local development environments.
