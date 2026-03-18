# Gaussian Fast Path

This is the most important implementation page for understanding `hsplat-warp`.

The Gaussian fast path is where the project prepares a batched Gaussian representation, converts it into a Fourier-domain angular spectrum, and then dispatches the accumulation kernel to either the native CUDA extension or the Warp backend.

## Entry point

The path starts when:

- `cfg.method == "naive_fast"`
- the loaded primitive container is `Gaussians`

The call chain is:

1. `get_cgh_method(cfg)`
2. `naive_cgh_from_primitives_fast(primitives, cfg)`
3. `cgh_from_primitives_fast(primitives, cfg)`
4. `cgh_from_gaussians_fast(primitives, cfg)`
5. `fully_analytic_cgh_gaussians_fast(primitives, cfg)`

## What `fully_analytic_cgh_gaussians_fast(...)` does

This function is the true implementation core of the fast path.

### Step 1: build the frequency grid

The function starts by building:

- `fx`
- `fy`
- `fz`

using `utils.make_freq_grid(cfg)`.

These represent the sampled spatial-frequency coordinates used to build the angular spectrum.

### Step 2: convert Gaussian orientation and scale into local transforms

From `primitives.quats` and `primitives.scales`, the code builds:

- `Rx`, `Ry`, `Rz`
- the combined rotation `R`
- the 2D local transform `A`
- `A_inv_T`
- `A_det`

This is the part that converts the Gaussian parameterization into the local reference frame used by the analytic spectrum computation.

### Step 3: derive per-Gaussian carrier and shift terms

The function also computes:

- `c`
- `du`
- `local_AS_shift`

These values encode how each Gaussian is shifted, rotated, and carrier-modulated in frequency space before accumulation.

### Step 4: choose the backend

Backend selection is resolved through `_resolve_gaussian_backend(cfg)`.

The result is one of:

- `cuda_ext`
- `warp`
- `fallback`

`fallback` is handled earlier in `naive_cgh_from_primitives_fast(...)`, where the code warns and drops back to `naive_slow`.

### Step 5: launch the accumulation kernel

The selected backend receives the same batched tensors:

- `fx`, `fy`, `fz`
- `R`
- `A_inv_T`
- `A_det`
- `c`
- `du`
- `local_AS_shift`
- `opacities`
- `colors`

The backend returns:

- `G_real`
- `G_imag`

Together, these form the Fourier-domain representation `G`.

### Step 6: transform back to the spatial wavefront

After kernel accumulation:

1. `G_real + 1j * G_imag` forms the complex spectrum
2. the code runs inverse FFT
3. the result is cropped to `cfg.resolution_hologram`
4. a pixel-pitch normalization is applied
5. thresholding removes very small residual values

The returned value is the final complex wavefront for the fast Gaussian path.

## Native CUDA vs Warp split point

The split happens only at the accumulation kernel boundary.

That means:

- tensor preparation is shared
- inverse FFT reconstruction is shared
- output normalization is shared

Only the accumulation implementation changes:

- native extension path: `hsplat/cuda/_wrapper.py` -> C++/CUDA extension
- Warp path: `hsplat/warp_backend.py`

This is why the Warp integration remains low-risk while still being meaningful.

## Warp backend behavior

`hsplat/warp_backend.py` mirrors the same kernel signature and accumulates into `G_real` and `G_imag`.

Important details:

- the backend expects float32 tensors on a single device
- it initializes Warp lazily
- it assigns a writable cache directory when `WARP_CACHE_DIR` is unset
- it uses PyTorch tensors directly at the call boundary

In other words, Warp is replacing the custom kernel implementation, not the entire Gaussian fast path.

## Why this page matters

If you want to:

- compare CUDA and Warp outputs
- modify Gaussian accumulation logic
- add a new backend
- benchmark a new kernel design

this is the code path you should study first.
