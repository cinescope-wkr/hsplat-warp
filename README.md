# hsplat

An open-source library for computer-generated holography using primitives.

[Project Page](https://bchao1.github.io/gaussian-wave-splatting/) | [Paper](https://dl.acm.org/doi/10.1145/3731163) | [Technical Documentation](DOCUMENTATION.md)

## Fork Notice
> [!NOTE]
> This repository is a fork of the original `hsplat` project. Implementation-level documentation for the `hsplat` repository structure is provided in [DOCUMENTATION.md](DOCUMENTATION.md).
>
> **Fork maintainer**: [Jinwoo Lee](cinescope-wkr.github.io) (cinescope@kaist.ac.kr)

**Changes in this fork**:
- `pycolmap` parser compatibility update for recent API versions
  (`SceneManager` import path replaced with `Reconstruction`-based loading).
- `gsplat.spherical_harmonics` mask-shape compatibility fix in 2DGS loading
  (radii from `[C, N, 2]` collapsed to boolean visibility mask `[C, N]`).
- `Getting Started` updates for this fork
  (`requirements.txt` workflow and recommended version combination).

## Associated Paper
#### Gaussian wave splatting for computer-generated holography | SIGGRAPH 2025
<img src="gws-teaser.png" width="100%">

[Project Page](https://bchao1.github.io/gaussian-wave-splatting/) | [Paper](https://dl.acm.org/doi/10.1145/3731163)

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
Use Python 3.10+ and install dependencies required by `hsplat`:
```bash
pip install -r requirements.txt
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

### 3) Data and checkpoints
Download [Mip-NeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) and
[NeRF synthetic](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4), then place datasets in `hsplat/data`.

Place pretrained Gaussian checkpoints in `hsplat/models` (example path:
`hsplat/models/blender_default/lego/10000/ckpts/ckpt_29999.pt`).
Pre-optimized checkpoints are available [here](https://drive.google.com/drive/folders/1zLCgHprvcwg1pDRiqiARcrXwHu4tWhAW?usp=drive_link).

### 4) Quick run
Run from `hsplat/hsplat`:
```bash
cd hsplat
bash scripts/main_gws_light.sh
```

## Overview
The code is organized as follows:
- The ```dsplat``` folder contains the phase encoding function (e.g., DPAC) for the SLMs, from the complex-valued wavefront output of ```hsplat```.
- The ```gsplat``` folder contains the [gsplat](https://github.com/nerfstudio-project/gsplat) library.
- The ```hsplat``` folder contains CGH algorithm implementations using primitives.

## Running the Code

To run our Gaussian Wave Splatting algorithm, run:
``` 
bash scripts/main_gws_light.sh
```

To run CGH algorithms for other primitives, run 
``` 
bash scripts/main_pc.sh  # point cloud
bash scripts/main_meshes.sh  # polygon-based CGH
bash scripts/main_gws.sh  # GWS matcinhg number of primitives
```

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

If you have any questions, please feel free to email [Suyeon Choi](https://choisuyeon.github.io/) and [Brian Chao](https://bchao1.github.io/).
