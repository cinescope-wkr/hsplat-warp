# Experiments and Scripts

The shell scripts under `hsplat/scripts/` are thin wrappers around `main.py`.

## Main scripts

| Script | Purpose |
| --- | --- |
| `main_gws.sh` | Full Gaussian wave-splatting experiments |
| `main_gws_light.sh` | Lighter Gaussian run with explicit output paths |
| `main_meshes.sh` | Polygon CGH over textured meshes |
| `main_pc.sh` | Point-cloud CGH from mesh-derived sampling |

## Backend-related usage

When using the Gaussian fast path, you can explicitly select the backend:

```bash
cd hsplat
python main.py --method naive_fast --gaussian-backend auto
python main.py --method naive_fast --gaussian-backend cuda_ext
python main.py --method naive_fast --gaussian-backend warp
```

## Suggested workflow for backend experiments

1. Start with `gaussian_backend=auto`.
2. Use `cuda_ext` and `warp` explicitly when comparing outputs or runtime behavior.
3. Keep the rest of the pipeline fixed while evaluating backend changes.
