# hsplat-warp

`hsplat-warp` is a Warp-extended fork of `hsplat` for primitive-based computer-generated holography research.

This documentation site is the new front door for the fork. The repository still keeps the on-disk package layout `hsplat/` and `dsplat/` for compatibility, but the project should be understood and referenced as `hsplat-warp`.

## What makes this fork different

- It keeps the original `hsplat` pipeline and scripts compatible.
- It adds an optional [NVIDIA Warp](https://github.com/NVIDIA/warp) backend for the Gaussian `naive_fast` path.
- It emphasizes maintainability and research extensibility instead of treating custom kernels as one-off implementation details.

## Start here

- New users: go to [Getting Started](getting-started.md)
- Contributors: go to [Architecture Overview](architecture/index.md)
- Kernel developers: go to [Native and Warp Backends](architecture/backends.md)
- Researchers comparing fork direction: go to [Why hsplat-warp](research/why-hsplat-warp.md)

## Implementation guides

- End-to-end runtime flow: [Execution Flow](architecture/execution-flow.md)
- Method differences: [Algorithms](architecture/algorithms.md)
- Batched Gaussian kernel path: [Gaussian Fast Path](architecture/gaussian-fast-path.md)
- Angular spectrum propagation details: [Propagation](architecture/propagation.md)

## Documentation structure

- `README.md` remains the concise repository landing page.
- This MkDocs site is the main structured documentation entry point.
