# Getting Started

## Clone

```bash
git clone https://github.com/cinescope-wkr/hsplat.git
cd hsplat
git submodule update --init --recursive
```

## Python environment

Use Python 3.10+ and install the core runtime dependencies:

```bash
pip install -r requirements.txt
```

Recommended version combination:

- `Python 3.10`
- `torch 2.9.1`
- `pytorch3d 0.7.9`
- `gsplat 1.5.3`

Optional documentation dependency:

```bash
pip install -r requirements-docs.txt
```

Optional Warp backend dependency:

```bash
pip install -r requirements-warp.txt
```

## Quick run

Run from the repository's `hsplat/` package directory:

```bash
cd hsplat
bash scripts/main_gws_light.sh
```

## Useful documentation paths

- Pipeline overview: [Architecture / Pipeline](architecture/pipeline.md)
- Acceleration details: [Architecture / Native and Warp Backends](architecture/backends.md)
- Experiment wrappers: [Research Guide / Experiments and Scripts](research/experiments.md)

## Building the docs locally

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Then open `http://127.0.0.1:8000/`.
