# Native and Warp Backends

## Why this split exists

The `hsplat-warp` fork keeps most of the pipeline in PyTorch and only swaps the custom Gaussian fast accumulation kernel backend.

That design keeps the integration low-risk:

- FFT propagation remains in PyTorch
- scene loading remains in the existing project stack
- some loading paths still depend on `gsplat`
- only the custom hot path gets an alternative implementation

## Backend choices

The Gaussian fast path now supports:

- `gaussian_backend="auto"`
- `gaussian_backend="cuda_ext"`
- `gaussian_backend="warp"`

Resolution behavior:

- `auto` prefers the existing CUDA extension
- if the CUDA extension is unavailable, `auto` uses Warp when available
- if neither accelerated backend is available, `naive_fast` falls back to `naive_slow` with a warning

## Native CUDA extension

The existing path under `hsplat/cuda/` provides:

- PyTorch extension loading in `_backend.py`
- Python wrapper entry points in `_wrapper.py`
- C++ and CUDA bindings under `csrc/`

This remains the first-choice backend in `auto` mode when available.

## Warp backend

The new backend in `hsplat/warp_backend.py` provides:

- a Warp port of the Gaussian spectrum accumulation stage
- PyTorch tensor interop at the backend boundary
- a writable temp cache fallback when `WARP_CACHE_DIR` is unset

This backend exists primarily to improve:

- maintainability of custom kernels
- research extensibility for future primitive or accumulation experiments
- safer iteration compared with editing PyTorch C++/CUDA extension code directly

## Practical guidance

- Use `auto` for the safest default behavior.
- Use `warp` when actively developing or experimenting with the Warp kernel.
- Use `cuda_ext` when you want to force the legacy native path for comparison or regression testing.
