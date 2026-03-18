# hsplat-warp

`hsplat-warp` is a [NVIDIA Warp](https://github.com/NVIDIA/warp)-extended fork of `hsplat` for primitive-based
computer-generated holography research.

The codebase still keeps the package/module path `hsplat` on disk for compatibility,
but this fork should be understood and referenced as `hsplat-warp` in documentation.

[Project Page](https://bchao1.github.io/gaussian-wave-splatting/) | [Paper](https://dl.acm.org/doi/10.1145/3731163) | [Docs](https://cinescope-wkr.github.io/hsplat-warp/)

<img src="gws-teaser.png" width="100%">

## Quick Navigation

- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Repository Layout](#repository-layout)
- [NVIDIA Warp Backend](#nvidia-warp-backend)
- [Citation](#citation)
- [Contact](#contact)

## Fork Notice
> [!NOTE]
> This repository is the `hsplat-warp` fork of the original `hsplat` project.
> The main structured documentation lives at [cinescope-wkr.github.io/hsplat-warp](https://cinescope-wkr.github.io/hsplat-warp/).
>
> **Fork maintainer**: [Jinwoo Lee](cinescope-wkr.github.io) (cinescope@kaist.ac.kr)

`hsplat-warp` emphasizes:
- safer optional acceleration paths
- better long-term maintainability for custom kernels
- stronger extensibility for future researchers building on the fast Gaussian path

**Changes in this fork**:
- `pycolmap` parser compatibility update for recent API versions
  (`SceneManager` import path replaced with `Reconstruction`-based loading).
- `gsplat.spherical_harmonics` mask-shape compatibility fix in 2DGS loading
  (radii from `[C, N, 2]` collapsed to boolean visibility mask `[C, N]`).
- `Getting Started` updates for this fork
  (`requirements.txt` workflow and recommended version combination).
- [NVIDIA Warp](https://github.com/NVIDIA/warp) backend added for the Gaussian `naive_fast` kernel path.
  Motivation: make kernel iteration safer and easier for future researchers while
  preserving the original CUDA-extension path as the default-compatible option.

**Naming note**
- Project/fork name: `hsplat-warp`
- Current package/directory name: `hsplat`
- Reason: preserve compatibility with the upstream layout, imports, scripts, and paths

## Associated Paper
#### Gaussian wave splatting for computer-generated holography | SIGGRAPH 2025

[Suyeon Choi*](https://choisuyeon.github.io/), [Brian Chao*](https://bchao1.github.io/), Jacqueline Yang, [Manu Gopakumar](https://manugopa.github.io/), [Gordon Wetzstein](https://web.stanford.edu/~gordonwz/)  
*denotes equal contribution

## Getting Started
### 1) Clone and submodules
```bash
git clone https://github.com/cinescope-wkr/hsplat.git
cd hsplat
git submodule update --init --recursive
```

### 2) Environment
Use Python 3.10+ and install dependencies required by `hsplat-warp`:
```bash
pip install -r requirements.txt
```

Optional documentation tooling:
```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Recommended version combination:
- `Python 3.10`
- `torch 2.9.1` (CUDA 12.8 build)
- `pytorch3d 0.7.9`
- `gsplat 1.5.3`

Pinned versions in `requirements.txt` reflect a known working environment.

- `torch` (CUDA build recommended for GPU execution)
- `numpy`, `matplotlib`, `imageio`, `tyro`, `rich`
- `pytorch3d`
- `trimesh` (for mesh/point sampling paths)
- `pycolmap` (for COLMAP parser)
- `gsplat` (for 2DGS loading and rendering)

Optional accelerated backend:
- `warp-lang` for the [NVIDIA Warp](https://github.com/NVIDIA/warp) Gaussian kernel backend
- install with `pip install -r requirements-warp.txt`
- this backend is optional; the existing CUDA extension remains supported

### 3) Data and checkpoints
Download [Mip-NeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) and
[NeRF synthetic](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4), then place datasets in `hsplat/data`.

Place pretrained Gaussian checkpoints in `hsplat/models` (example path:
`hsplat/models/blender_default/lego/10000/ckpts/ckpt_29999.pt`).
Pre-optimized checkpoints are available [here](https://drive.google.com/drive/folders/1zLCgHprvcwg1pDRiqiARcrXwHu4tWhAW?usp=drive_link).

### 4) Quick run
Run from the repo’s `hsplat/` package directory:
```bash
cd hsplat
bash scripts/main_gws_light.sh
```

## Documentation

- `README.md`: concise repository overview and quick start
- `docs/`: MkDocs-based structured documentation for `hsplat-warp`
- hosted docs: [cinescope-wkr.github.io/hsplat-warp](https://cinescope-wkr.github.io/hsplat-warp/)

## Repository Layout

The repository is organized as follows:
- The `dsplat` folder contains the phase encoding function (e.g., DPAC) for the SLMs, from the complex-valued wavefront output of `hsplat-warp`.
- The `gsplat` folder contains the [gsplat](https://github.com/nerfstudio-project/gsplat) library.
- The `hsplat` folder contains the core implementation used by `hsplat-warp`.

## Additional Run Scripts

The main quick-start command is shown above in Getting Started. Additional experiment scripts include:

```bash
bash scripts/main_pc.sh  # point cloud
bash scripts/main_meshes.sh  # polygon-based CGH
bash scripts/main_gws.sh  # GWS matcinhg number of primitives
```

## NVIDIA Warp Backend
`hsplat-warp` supports an [NVIDIA Warp](https://github.com/NVIDIA/warp) backend for `method=naive_fast`.

Why this exists:
- Improve maintainability by moving custom-kernel development closer to Python.
- Improve research extensibility by making the Gaussian fast kernel easier to modify,
  prototype, and compare against the legacy CUDA extension.
- Keep the integration low-risk by limiting [Warp](https://github.com/NVIDIA/warp) to the custom Gaussian accumulation
  kernel instead of rewriting the whole pipeline.

Why it matters:
- It gives future contributors a clearer path to extend the fast Gaussian renderer.
- It reduces coupling between research iteration and handwritten PyTorch C++/CUDA bindings.
- It preserves backward compatibility because the existing CUDA extension still works
  and remains the first choice in `gaussian_backend=auto` when available.

Why this is part of the fork identity:
- It is the clearest architectural distinction between upstream `hsplat` and `hsplat-warp`.
- It signals that this fork is aimed at maintainable research iteration, not only one-off reproduction.
- It gives collaborators an obvious place to add future kernel experiments without replacing the full pipeline.

- Default: `gaussian_backend=auto`
- Explicit backends: `gaussian_backend=cuda_ext` or `gaussian_backend=warp`
- Safety behavior:
  if `auto` is selected and neither accelerated backend is available, `naive_fast`
  falls back to `naive_slow` with a warning instead of crashing.

Example:
```bash
cd hsplat
python main.py --method naive_fast --gaussian-backend warp
```

Notes:
- [Warp](https://github.com/NVIDIA/warp) is only used for the custom Gaussian accumulation kernel.
- FFT propagation and most orchestration stay in PyTorch.
- `gsplat` is still used in data-loading paths where applicable.
- If `WARP_CACHE_DIR` is not set, the backend uses a writable temp cache directory by default.
- [Warp](https://github.com/NVIDIA/warp) installation and driver requirements follow the [official installation guide](https://nvidia.github.io/warp/user_guide/installation.html).

## Citation

If you find our work useful in your research, please cite:

```
@article{choi2025gaussian,
  title={Gaussian wave splatting for computer-generated holography},
  author={Choi, Suyeon and Chao, Brian and Yang, Jacqueline and Gopakumar, Manu and Wetzstein, Gordon},
  journal={ACM Transactions on Graphics (TOG)},
  volume={44},
  number={4},
  pages={1--13},
  year={2025},
  publisher={ACM New York, NY, USA}
}
```

## Contact

For `hsplat-warp` issues, documentation, and fork-specific backend questions, please contact [Jinwoo Lee](cinescope-wkr.github.io).

For questions about the original paper and the upstream `hsplat` method, please refer to the original authors, including [Suyeon Choi](https://choisuyeon.github.io/) and [Brian Chao](https://bchao1.github.io/).
