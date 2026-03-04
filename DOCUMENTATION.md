# Technical Documentation: `hsplat` + `dsplat`

This companion document summarizes the structure, API surface, and execution semantics of the `hsplat-main` codebase from an implementation perspective.

## Quick Navigation
- [1) Scope](#scope)
- [2) Repository map](#repository-map)
- [3) Top-level entry points and control flow](#entry-points)
- [4) Data model and pipeline composition](#data-model)
- [5) Algorithms and rendering math](#algorithms)
- [6) Wave propagation](#wave-propagation)
- [7) Utility layer](#utility-layer)
- [8) Viz utilities](#viz-utilities)
- [9) Native / CUDA path](#native-cuda)
- [10) DPAC encoding](#dpac-encoding)
- [11) Scripted experiment entry points](#scripts)
- [12) Execution flow summary](#execution-flow)
- [13) Known caveats / maintenance notes](#caveats)
- [14) Connection to the paper](#paper-connection)

<a id="scope"></a>
## 1) Scope
This document covers:
- `hsplat-main/hsplat`: core CGH pipeline (primitives, data loading, propagation, algorithm dispatch, utilities)
- `hsplat-main/dsplat/main_dpac_encoding.py`: downstream phase encoding for SLM calibration
- CUDA extension surfaces used by both `hsplat` and optional acceleration paths
- Script entry points in `hsplat-main/hsplat/scripts`

<a id="repository-map"></a>
## 2) Repository map
- `hsplat-main/README.md`: top-level setup, [paper references](https://dl.acm.org/doi/10.1145/3731163), and run commands
- `hsplat-main/hsplat/README.md`: package-level overview and usage notes
- `hsplat-main/hsplat/main.py`: primary execution pipeline
- `hsplat-main/hsplat/load_data.py`: dataset/model ingestion and primitive construction
- `hsplat-main/hsplat/primitives.py`: primitive data model and operations
- `hsplat-main/hsplat/algorithms.py`: CGH algorithm implementations
- `hsplat-main/hsplat/propagations.py`: propagation operators
- `hsplat-main/hsplat/utils.py`: numerical and geometry helper utilities
- `hsplat-main/hsplat/viz_utils/{parser.py,visualization.py,normalize.py,__init__.py}`: scene parsing and camera path utilities
- `hsplat-main/hsplat/cuda/{_backend.py,_wrapper.py,csrc/*}`: native extension loading and kernels
- `hsplat-main/dsplat/main_dpac_encoding.py`: phase encoding and LUT workflow

---

<a id="entry-points"></a>
## 3) Top-level entry points and control flow

### 3.1 Main CGH entry: `hsplat/main.py`
`main.py` wires together:
1. config parsing (`tyro.cli` into `Config`)
2. scene/camera loading (`get_camera_params`)
3. primitive loading (`load_data.load_primitives`)
4. algorithm dispatch (`algorithms.get_cgh_method`)
5. optional focal sweep and saving (`ASM_parallel`, `save_results`)

`if __name__ == "__main__"` executes per-frame and per-channel loops for static and trajectory datasets.

### 3.2 DPAC/SLM encoding entry: `dsplat/main_dpac_encoding.py`
`main(cfg)` takes a wavefront (`cfg.wavefront_path`) and produces phase-only LUT-based SLM output (`cfg.out_path`) with optional per-pixel LUT and laser compensation.

---

<a id="data-model"></a>
## 4) Data model and pipeline composition

## 4.1 Data model (`primitives.py`)

### Factory helpers
- `get_obj_from_module(module, obj_name)`
- `get_module_from_obj_name(name)`
- `get_obj_by_name(name)`
- `call_func_by_name(*args, func_name=None, **kwargs)`
- `construct_class_by_name(*args, class_name=None, **kwargs)`

These helpers enable dynamic construction by string name and are generic utilities for dynamic imports.

### Base classes
- `Primitives(nn.Module)`
  - container semantics for heterogeneous primitive collections
  - methods: `add_primitive`, iterator/index access, length, `.to`, `select_random_points`, `sort` (abstract)
- `Primitive(nn.Module)`
  - base fields: `position`, `normal`, `amplitude`, `phase`
  - property `normal` normalizes tensor conversion
  - `shade_illumination(illu)` applies orientation-aware cosine shading
  - abstract: `get_wavefront_contribution`, `get_sh_color`
  - `to()` migrates core tensor fields

### Concrete primitive classes
- `Point(Primitive)`
  - scalar point primitive
  - fields: `color`, `opacity`, method `z()`, and color accessor
- `Polygon(Primitive)`
  - triangle primitive backed by `v1/v2/v3` and centroid
  - has computed `X` (3x3 vertices), `normal`, optional quaternion
  - method `compute_affine_matrix(canonical_points, actual_points)`
  - `center_z_coord` shifts centroid to z-zero plane
- `Gaussian(Primitive)`
  - gaussian primitive with SH features (`sh0`, `shN`), scale, quaternion
  - method `get_sh_color(direction)` evaluates SH up to degree 3 in torch
  - color path supports either color scalar coefficient or explicit SH
  - `to()` migrates all gaussian tensors
- `Gaussians(Primitives)`
  - batched tensor-backed gaussian set: `means, opacities, quats, scales, sh0, shN, colors, cov2ds`
  - vectorized APIs:
    - `sort(order='back2front'|'front2back'|'normal_z')`
    - `cull_elements(standard, threshold)` supports thresholds for depth, canvas bounds, angle, scale, bbox, etc.
    - `remap_depth_range(depth_range, sigma=3, clamp_normalized_depth=True, original_depth_range=None)`
    - `transform_perspective(K, pixel_pitch)` projects to pixel-space with covariance update
    - `set_scales`, `set_quats`, `flip_z`, `zero_z`, `sample_points`
- `Polygons(Primitives)`
  - triangle-batch representation as `[N,3,3]` vertex list
  - methods: `sort`, `cull_elements`, `remap_depth_range`, `transform_perspective`, `flip_z`, `zero_z`
- `Points(Primitives)`
  - point-batch representation
  - methods: `sort` via inherited, `cull_elements`, `remap_depth_range`, `set_zero_phase`, `transform_perspective`, `flip_z`, `zero_z`, `sample_points`, and property `z`

## 4.2 Data ingestion (`load_data.py`)

### Dispatch and common helpers
- `_get_device(dev)` — picks CUDA if available.
- `_visibility_mask_from_radii(radii, keep_batch_dim=False)` — converts gsplat radii to boolean visibility masks.
- `load_primitives(cfg)` — chooses one of:
  - `load_gaussians`
  - `load_textured_mesh`
  - `load_points_from_mesh`
  - `load_points_from_sfm`
  - `load_points_from_gaussians`

### Point/surface loaders
- `load_gaussians(pt_path, ...)`
  - loads gsplat checkpoint via `load_2dgs_ckpt`
  - optional ground-truth projection via `rasterization_2dgs`/`rasterization_2dgs_wsr`
  - transforms means/quaternions into view space
  - applies perspective transform
  - culls, samples, remaps depth, flips z, and returns target amplitude
- `load_textured_mesh(mesh_path, ...)`
  - loads `.npz` meshes + optional gsplat alpha-mask fallback
  - builds polygon primitives from faces
  - culls via gsplat radius, remaps depth, projects target amplitude
- `load_points_from_mesh(mesh_path, ...)`
  - samples points from mesh using `sample_from_mesh` (sampling via trimesh)
  - can output as `Gaussians` (default) or pixel points (`load_points_as_pixels=True`)
  - returns projected target and filtered/scaled points
- `load_points_from_sfm(sfm_path, ...)`
  - builds point primitives from COLMAP points/colors through `viz_utils.parser.COLMAPParser`
- `load_points_from_gaussians(pt_path, ...)`
  - loads raw gsplat gaussian samples and converts to un-oriented gaussian points
  - resets orientation to identity quaternions and returns projected target
- `orthographic_projection_2d(primitives, target_resolution, pixel_pitch, dev=None, alpha_blending=False, illum=None)`
  - renderer fallback for no-alpha and alpha compositing modes
- `generate_target_grid(target_resolution, dev, pixel_pitch)`
  - returns X/Y coordinate grids in object coordinate convention

### Mesh/point helper utilities
- `sample_from_mesh(vertices, faces, texture_map_idx, uvs, texture_maps, num_points=None)`
- `preselect_random_points(num_points, *args)`
- `load_point_cloud_from_ply(ply_path, ...)`
- `load_mesh_from_ply(ply_path, ..., output='default'|'polygons'|'points'|'gaussians', ...)`

---

<a id="algorithms"></a>
## 5) Algorithms and rendering math

## 5.1 `algorithms.py`

### Dispatch helpers
- `get_cgh_method(cfg)`
  - maps string mode to callable
  - supported: `naive_slow`, `naive_fast`, `silhouette`, `alpha-wave-blending`

### Core utilities
- `ifft(a)` and `fft(a)` for centered FFT/IFT with ortho normalization.
- `lateral_shift(input_wavefront, shift, pixel_pitch, resolution_hologram)`
  - uses affine grid sampling to spatially shift complex fields.

### Primitive-to-wavefront pathways
- `cgh_from_point(primitive, cfg)`
- `cgh_from_triangle(primitive, cfg, prev_wavefront=None)`
- `cgh_from_gaussian(primitive, cfg, prev_wavefront=None, frequency_grid=None)`
- `cgh_from_gaussians_fast(primitives, cfg, prev_wavefront=None)`
- `cgh_from_primitive(primitive, cfg, prev_wavefront=None, frequency_grid=None)`
- `cgh_from_primitives_fast(primitives, cfg, prev_wavefront=None)`

### Methods
- `naive_cgh_from_primitives(primitives, cfg)`
  - loop over primitive collection and accumulate field with occlusion mask `T`
- `naive_cgh_from_primitives_fast(primitives, cfg)`
  - vectorized/naive fast path using batch CUDA for gaussian groups
- `silhouette_based_method(primitives, cfg)`
  - order aware occlusion by propagation/back-propagation at per-primitive depth
- `alpha_wave_blending(primitives, cfg)`
  - front/back ordered alpha-compositing variant using phase compensation for each primitive

### Angular spectrum / analytic functions
- `angular_spectrum_reference(fx, fy, shape='triangle'|'rectangle'|'circle'|'gaussian')`
- `fully_analytic_cgh_basic(primitive, cfg, shape='triangle', return_angular_spectrum=False, ...)`
- `fully_analytic_cgh_gaussian(primitive, cfg, shape='gaussian', return_angular_spectrum=False, frequency_grid=None, ...)`
- `fully_analytic_cgh_gaussians_fast(primitives, cfg, ...)`

---

<a id="wave-propagation"></a>
## 6) Wave propagation (`propagations.py`)

### ASM_rotation(nn.Module)
- `forward(wavefront, rotation=None, return_angular_spectrum=False)`
  - performs FFT-based tilt-rotated propagation using padded domain
- `w_uv(uu, vv, wavelength)`
- `remap_angular_spectrum(angular_spectrum, rotation=None)`
- `fourier_coordinate_transform(u, v, T, wavelength, direction='ref2src')`
- properties: `rotation`, `H`
- `compute_H(...)` is abstract (`pass`)

### ASM_parallel(nn.Module)
- `forward(wavefront, z, linear_conv=True, phase_compensation=None)`
- `propagate(wavefront)` placeholder
- `compute_H(input_field, prop_dist, wavelength, pixel_pitch, lin_conv=True, ho=(1,1))`
  - builds transfer function with padded FFT aperture
- `prop(u_in, H, linear_conv=True, padtype='zero', ho=(1,1), phase_compensation=None)`
  - runs FFT, optional pre/post phase compensation and cropping

---

<a id="utility-layer"></a>
## 7) Utility layer (`utils.py`)

### Grid/image and tensor helpers
- `pad_image(field, target_shape, pytorch=True, stacked_complex=False, padval=0, mode='constant', lf=False)`
- `crop_image(field, target_shape, pytorch=True, stacked_complex=False, lf=False)`
- `grid_sample_complex(input_field, grid, align_corners=False, mode='bilinear', coord='rect')`

### Rotation/math
- `coordinate_rotation_matrix(axis, angle, keep_channel_dim=False)`
- `normalized_quat_to_rotmat(quat)`
- `quaternion_to_euler_angles(q)`
- `quaternion_to_euler_angles_zyx(q, flip_angles_for_large_x=True)`
- `rotation_matrix_to_quaternion(R)`
- `quaternion_multiply(q1, q2)`

### Frequency and transforms
- `make_freq_grid(cfg)`
- `rotate_frequency_grid(R, fx, fy, fz=None, wavelength=None)`
- `get_rotation_matrix(n)`

### SH and signal helpers
- `factorial(n)`
- `legendre_polynomial(l, m, x)`
- `compute_sh_basis(L, direction)`
- `get_gaussian_amplitude(fx, fy)`
- `im2float(im, dtype=np.float32, im_max=None)`
- `compute_quaternions_from_triangles(triangles)`
- `compute_scales_from_triangles(triangles)`

### I/O and visualization support
- `normalize_and_write(name, a, max_val=None, min_val=None)`
- `decode_dict_to_tensor(data, order='rgb')`
- `gsplat_projection_2dgs(means, quats, viewmats)`
- `save_video(amps, out_folder)`
- `save_focal_stack(amps, out_folder)`
- `get_intrinsics_keep_fov(K, width, height)`
- `get_intrinsics_resize_to_fit(K, width, height)`
- `scale_to_range(x, a, b)`
- `compute_barycentric_coords(point, triangle)`

### Constants
- `BLENDER_SCENES`, `MIPNERF360_SCENES`, `NUM_TRIANGLES`

---

<a id="viz-utilities"></a>
## 8) Viz utilities (`hsplat/viz_utils`)

### `parser.py`
- `_load_colmap_manager(colmap_dir)`
- `_get_rel_paths(path_dir)`
- `COLMAPParser(data_dir, factor=1, normalize=False, test_every=8)`
  - loads COLMAP reconstruction into cam/world/points arrays
  - builds and stores:
    - `camtoworlds`, `camtoworld` per view
    - `Ks_dict`, `params_dict`, `imsize_dict`, masks/distortion maps
    - `points`, `points_err`, `points_rgb`, `point_indices`
  - methods:
    - `get_image(idx)` (selected by normalized test frame spacing)
- `BlenderParser(data_dir, split='train')`
  - reads transforms json and images
  - method:
    - `get_image(idx)`

### `visualization.py` and `normalize.py`
Both files currently provide helpers with overlapping names and behavior:
- `get_focus_point(direction, origin)`
- `view_pose(lookdir, up)`
- `get_ellipse_path(cam2worlds, num_frames, ...)`
- `interpolate_path(cam2worlds, num_frames)`

Note: trajectory helpers are implemented in `visualization.py` and imported into `parser.py` for compatibility.

### `viz_utils/__init__.py`
- re-exports `parser` and `visualization`.

---

<a id="native-cuda"></a>
## 9) Native / CUDA path (`hsplat/cuda`)

### Python-side loader (`_backend.py`)
- `load_extension(...)` — wraps `torch.utils.cpp_extension.load`
- `cuda_toolkit_available()` and `cuda_toolkit_version()`
- Import strategy:
  1. tries `from hsplat import csrc as _C`
  2. fallback to JIT-compile `hsplat_cuda` from `.cu/.cpp` under `csrc/`
- exports `__all__ = ["_C"]`

### Python wrapper (`_wrapper.py`)
- `add(a, b)` for testing
- `cgh_gaussians_naive(...)` calls `_C.cgh_gaussians_naive`

### CUDA C++/CUDA interface
- `csrc/bindings.h`
  - declares `add_tensor` and `cgh_gaussians_naive_tensor`
- `csrc/ext.cpp`
  - `PYBIND11_MODULE` exports:
    - `add` -> `hsplat::add_tensor`
    - `cgh_gaussians_naive` -> `hsplat::cgh_gaussians_naive_tensor`
- `csrc/test.cu`
  - naive tensor addition kernel and wrapper
- `csrc/cgh_gaussians_naive.cu`
  - `cgh_gaussians_naive_kernel` (batched gaussian spectrum accumulation)
  - memory batching using shared memory
  - `cgh_gaussians_naive_tensor(...)` validates tensors, allocates output, launches kernel, synchronizes
- `csrc/CMakeLists.txt`
  - builds a Python module target `cuda_extension` from `.cu` and `.cpp`
- `csrc/CUDA version` currently includes helper header `utils.cuh` but file is empty (0 lines)

---

<a id="dpac-encoding"></a>
## 10) DPAC encoding (`dsplat/main_dpac_encoding.py`)

### Core phase encoding functions
- `naive_lut_phase_encoding(phase, max_phase=2*pi)`
  - maps 2D phase to `uint8` via wrapping + scaling
- `double_phase_encoding_multi_level(field, lut_perpixel=None, ... , max_phase=3*pi, ref_levels=None)`
  - amplitude->phase conversion
  - produces checkerboard phase-diff assignment for double-phase representation
  - optional LUT-guided phase difference
- `load_luts(lut_path, ref_levels=None)`
  - loads per-level LUT tensors and interpolates to `1024x1024`
- `get_per_pixel_lut_per_level(phase, lut_folder_path, calibrated_ref_levels, max_phase=3*pi)`
  - selects best per-pixel LUT among reference levels

### Config/dataclass
- `ConfigDPAC(Config)` extends `main.Config` with fields:
  - `lut_folder_path`, `lut_path`, `laser_amp_path`, `out_path`, `wavefront_path`, `max_phase`, `slm_res`

### `main(cfg)`
- reads wavefront
- crops/pads to SLM resolution
- optionally applies amplitude and laser compensation
- applies LUT selection and phase conversion
- writes encoded `uint8` PNG to output path

---

<a id="scripts"></a>
## 11) Scripted experiment entry points (`hsplat/scripts/*.sh`)
- `main_gws.sh`: Gaussian wave-splatting full match experiments for NeRF and Mip-NeRF scenes
- `main_gws_light.sh`: same family with explicit number-of-gaussian variant and output paths
- `main_meshes.sh`: polygon CGH over textured meshes
- `main_pc.sh`: point-cloud CGH from mesh-derived sampling

These scripts are thin CLI wrappers around `main.py` and set argument presets by scene and target asset.

---

<a id="execution-flow"></a>
## 12) Execution flow summary (recommended mental model)

For a Gaussian AWB run:
1. Parse config from CLI.
2. Resolve camera trajectory via `get_camera_params`.
3. `load_primitives` reads target checkpoint/mesh and returns typed primitive collection.
4. `process_cfg` sets resolution/pixel/depth, sampling, and defaults.
5. `get_cgh_method` picks strategy -> call returns complex wavefront.
6. Optional focal sweep + amplitude/phase extraction -> serialization and media output.
7. Optional DPAC post-pass in `dsplat` for SLM drive values.

---

<a id="caveats"></a>
## 13) Known caveats / maintenance notes
- Trajectory helpers are duplicated across parser/visualization compatibility paths; prefer importing from `visualization.py` directly in new code.
- `cgh_gaussians_naive.cu` hardcodes `batch_size=200` for shared-memory tiling; this should be validated against target architecture.
- Several modules contain mixed assumptions (e.g., orientation and depth sign conventions around z-flip and camera conventions) that are project-specific.
- `load_mesh_from_ply` `output='gaussians'` builds `data_dict['sh0']=torch.ones_like(faces)` which appears dimensionally inconsistent for `Gaussians` expecting `[N]` or `[N, ...]` SH coefficient shape in typical use.
- The parser/visualization utilities are split over `normalize.py` and `visualization.py` with overlapping names; avoid importing both with `*` or renaming ambiguously.

<a id="paper-connection"></a>
## 14) Connection to the paper ([ACM DOI](https://dl.acm.org/doi/10.1145/3731163))

The implementation follows the pipeline described in the paper as a staged mapping from scene representation to wavefront synthesis:

- Scene/primitives represent the paper’s object model: points, triangles, and Gaussians, with `Primitives`/`Gaussians` carrying position, orientation, extent, opacity, and SH color terms used by the wave-splat model.
- `load_data.load_primitives` and `load_*` helpers implement the data preparation path used by the method setup (scene ingest, canonicalization, projection, culling, and depth normalization) before any optical propagation.
- `algorithms.get_cgh_method` + `main` implement the three rendering modes discussed in the method section: naive point/triangle accumulation, silhouette-aware occlusion handling, and Alpha Wave Blending (AWB), with the AWB path explicitly selecting phase/opacity-composited ordering.
- `fully_analytic_*`/`cgh_from_*` paths map directly to the analytic primitive-to-hologram kernels (especially the Gaussian closed-form path), while `naive`/`ASM` paths provide fallbacks and debugging equivalents.
- `propagations.ASM_parallel` and `ASM_rotation` realize the propagation model, transfer-function construction, and optional compensated shift/tilt operations used in the forward model.
- `dsplat.main_dpac_encoding.main` encodes the final complex field into per-pixel SLM drive values and aligns with the paper’s DPAC/SLM calibration workflow described for deployment and hardware constraints.
- CUDA extension (`cgh_gaussians_naive`) is an optimization path for batching the Gaussian summation primitive kernel and is the main compute-speed branch for production inference.
