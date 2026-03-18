# Pipeline

## Top-level control flow

The main CGH path is centered in `hsplat/main.py`:

1. `Config` is parsed from the CLI.
2. `process_cfg()` derives runtime parameters such as `resolution_hologram`.
3. `get_camera_params()` loads the camera path and intrinsics.
4. `load_primitives(cfg)` loads scene data into `Points`, `Polygons`, or `Gaussians`.
5. `get_cgh_method(cfg)` chooses the rendering method.
6. The resulting wavefront is optionally propagated across focal slices and saved.

## Data model

The primitive hierarchy in `hsplat/primitives.py` is the central abstraction layer:

- `Primitive` stores `position`, `normal`, `amplitude`, and `phase`
- `Point`, `Polygon`, and `Gaussian` represent the individual primitive types
- `Points`, `Polygons`, and `Gaussians` provide batched container variants

The Gaussian batched path is particularly important because it is the only part of the current core with a dedicated accelerated backend.

## Data ingestion

`hsplat/load_data.py` handles:

- Gaussian checkpoint loading
- mesh-to-polygon and mesh-to-point conversion
- COLMAP/SfM point loading
- target amplitude generation
- culling, depth remapping, and perspective transforms

The project also depends on `gsplat` CUDA wrappers in some loading paths, so the custom Warp backend does not replace the entire GPU stack.

## Rendering and propagation

`hsplat/algorithms.py` contains:

- method dispatch
- per-primitive analytic CGH functions
- the Gaussian fast path
- occlusion-aware modes such as silhouette and alpha-wave-blending

`hsplat/propagations.py` contains the angular spectrum propagation operators used after wavefront formation.

To understand the actual runtime flow in more detail, continue to:

- [Execution Flow](execution-flow.md)
- [Algorithms](algorithms.md)
- [Gaussian Fast Path](gaussian-fast-path.md)
- [Propagation](propagation.md)
