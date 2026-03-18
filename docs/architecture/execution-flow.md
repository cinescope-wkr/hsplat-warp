# Execution Flow

This page explains the main runtime path of `hsplat-warp` as it is implemented today, using the actual control flow in `hsplat/main.py`.

## 1. CLI config enters through `main.py`

The executable entry point is `hsplat/main.py`.

At startup:

1. `tyro.cli(Config)` parses the command line into a `Config` dataclass.
2. `process_cfg(cfg)` derives runtime parameters such as:
   - `resolution_hologram`
   - `pad_n`
   - point-sampling defaults for point-based assets
3. `get_camera_params(cfg)` loads camera trajectories and intrinsics.

This means the configuration object is the central runtime contract for the rest of the pipeline.

## 2. Scene loading happens inside `main(cfg)`

`main(cfg)` performs three main tasks:

1. `load_primitives(cfg)` loads the target scene representation.
2. `get_cgh_method(cfg)` chooses the rendering method.
3. The selected method returns a complex wavefront tensor.

`load_primitives(cfg)` can return different primitive containers, depending on `cfg.target_asset`:

- `Gaussians`
- `Polygons`
- `Points`

At the same time, it also returns `target_amp`, which is used later for normalization and saved outputs.

## 3. Rendering happens per frame and per channel

The top-level `__main__` block loops over:

- camera frames
- color channels
- repeated samples when `cfg.num_frames > 1`

For each channel:

1. `cfg.wavelength` is set from `cfg.wavelengths`
2. `cfg.illum` is set
3. `main(cfg)` is called
4. the resulting wavefronts are stacked
5. `save_results(...)` writes wavefront-derived amplitude and angle outputs

After the wavefront is formed, the code also performs a focal sweep by repeatedly applying `ASM_parallel(cfg)` to the saved wavefront.

## 4. Method dispatch is late and explicit

`get_cgh_method(cfg)` in `hsplat/algorithms.py` maps the configured method name to one of:

- `naive_cgh_from_primitives`
- `naive_cgh_from_primitives_fast`
- `silhouette_based_method`
- `alpha_wave_blending`

This late binding is useful for research because the loading path and output path remain mostly unchanged while only the rendering method changes.

## 5. Output formation and propagation are separate stages

It helps to think of the pipeline in two stages:

### Stage A: primitive to wavefront

This stage is handled in `hsplat/algorithms.py`.

Its job is to convert the scene primitives into a complex wavefront at the hologram plane or object depth, depending on the method.

### Stage B: wavefront propagation and saving

This stage is handled by:

- `ASM_parallel` in `hsplat/propagations.py`
- `save_results(...)` in `hsplat/main.py`
- video and focal-stack helpers in `hsplat/utils.py`

This separation is important because the custom Warp and CUDA backends only affect Stage A, not the full pipeline.

## 6. Where most backend work matters

The most important accelerated path today is:

- `method="naive_fast"`
- `target_asset="gaussians"`

That path eventually routes into the batched Gaussian implementation in `fully_analytic_cgh_gaussians_fast(...)`.

For the exact details of that path, continue to [Gaussian Fast Path](gaussian-fast-path.md).
