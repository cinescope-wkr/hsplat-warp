import torch
from torch import Tensor

from ._backend import _C


def cuda_extension_available() -> bool:
    return _C is not None


def cuda_extension_unavailable_reason() -> str:
    return (
        "hsplat CUDA extension is unavailable. Build it with a CUDA toolkit "
        "visible to PyTorch, or select gaussian_backend=warp."
    )


# simple CUDA test function
def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _C is None:
        raise RuntimeError(cuda_extension_unavailable_reason())
    return _C.add(a, b)

### BEGIN FAST CGH FUNCTIONS

def cgh_gaussians_naive(
    fx             : torch.Tensor,
    fy             : torch.Tensor,
    fz             : torch.Tensor,
    wvl            : float,
    R              : torch.Tensor,
    A_inv_T        : torch.Tensor,
    A_det          : torch.Tensor,
    c              : torch.Tensor,
    du             : torch.Tensor,
    local_AS_shift : torch.Tensor,
    opacity        : torch.Tensor,
    color          : torch.Tensor
) -> torch.Tensor:
    if _C is None:
        raise RuntimeError(cuda_extension_unavailable_reason())
    return _C.cgh_gaussians_naive(
        fx, 
        fy, 
        fz, 
        wvl, 
        R, 
        A_inv_T, 
        A_det, 
        c, 
        du, 
        local_AS_shift,
        opacity,
        color
    )
