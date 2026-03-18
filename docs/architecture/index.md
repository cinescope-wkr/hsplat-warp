# Architecture Overview

`hsplat-warp` keeps the original `hsplat` project structure, but re-frames it around a clearer fork identity and an optional Warp acceleration path.

## Repository map

| Path | Role |
| --- | --- |
| `README.md` | Top-level setup, fork identity, and quick usage |
| `hsplat/main.py` | Primary CGH execution pipeline |
| `hsplat/load_data.py` | Dataset/model ingest and primitive construction |
| `hsplat/primitives.py` | Primitive data model and operators |
| `hsplat/algorithms.py` | CGH algorithm implementations and backend selection |
| `hsplat/propagations.py` | Propagation operators |
| `hsplat/utils.py` | Numerical and geometry helpers |
| `hsplat/cuda/*` | Native extension loading and CUDA kernels |
| `hsplat/warp_backend.py` | Optional Warp Gaussian fast-path backend |
| `dsplat/main_dpac_encoding.py` | Downstream phase encoding for SLM calibration |

## Main execution flow

`hsplat/main.py` wires together:

1. config parsing with `tyro`
2. scene and camera loading
3. primitive loading
4. CGH algorithm dispatch
5. optional focal sweep and result saving

For the most relevant execution details, continue to [Pipeline](pipeline.md).

For implementation-level walkthroughs, also see:

- [Execution Flow](execution-flow.md)
- [Algorithms](algorithms.md)
- [Gaussian Fast Path](gaussian-fast-path.md)
- [Propagation](propagation.md)
