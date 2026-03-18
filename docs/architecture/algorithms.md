# Algorithms

This page explains how the main CGH methods differ in implementation terms.

## Method dispatch

`get_cgh_method(cfg)` selects the rendering function based on `cfg.method`.

The currently supported methods are:

- `naive_slow`
- `naive_fast`
- `silhouette`
- `alpha_wave_blending`

## `naive_slow`

`naive_cgh_from_primitives(...)` loops over primitives one by one.

For each primitive:

1. it calls `cgh_from_primitive(...)`
2. thresholds very small amplitudes
3. applies opacity and color
4. accumulates into the output wavefront

This path is simple and useful as a correctness-oriented fallback, but it is not the preferred path for large Gaussian sets.

## `naive_fast`

`naive_cgh_from_primitives_fast(...)` is the main accelerated path.

Its key properties:

- only supports the batched `Gaussians` container
- resolves the backend through `_resolve_gaussian_backend(cfg)`
- uses either:
  - the native CUDA extension
  - the Warp backend
  - or, in `auto`, falls back to `naive_slow` if neither accelerated backend exists

This is the core method where `hsplat-warp` differs most clearly from upstream-style maintenance.

## `silhouette`

`silhouette_based_method(...)` uses depth ordering and repeated propagation to account for occlusion.

Its structure is:

1. sort primitives back-to-front
2. propagate the current wavefront to the primitive depth
3. compute the primitive contribution
4. combine the propagated background and the new object contribution
5. propagate back toward the hologram plane

Compared with `naive_slow`, this method spends more effort on occlusion-aware composition.

## `alpha_wave_blending`

`alpha_wave_blending(...)` is another occlusion-aware method.

Its structure is:

1. sort primitives front-to-back or back-to-front
2. compute each primitive wavefront
3. optionally add random phase
4. propagate the primitive contribution to the SLM plane
5. update the accumulated transmittance mask `T`

This path is more compositing-oriented than `silhouette`, and it depends on phase compensation behavior for the propagated object wavefront.

## Shared primitive dispatch

All methods ultimately rely on some combination of:

- `cgh_from_point(...)`
- `cgh_from_triangle(...)`
- `cgh_from_gaussian(...)`
- `cgh_from_gaussians_fast(...)`

That means the top-level methods differ mainly in:

- composition strategy
- ordering
- propagation timing
- backend choice for the Gaussian fast path
