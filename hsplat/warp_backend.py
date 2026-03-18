import logging
import os
import tempfile

import torch

logger = logging.getLogger(__name__)

_WARP_IMPORT_ERROR = None

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - import guard only
    wp = None
    _WARP_IMPORT_ERROR = exc


if wp is not None:
    _PI = 3.14159265358979323846
    _TWO_PI = 2.0 * _PI
    _WARP_INITIALIZED = False

    @wp.kernel
    def _cgh_gaussians_naive_kernel(
        fx: wp.array2d(dtype=wp.float32),
        fy: wp.array2d(dtype=wp.float32),
        fz: wp.array2d(dtype=wp.float32),
        wvl: float,
        R: wp.array3d(dtype=wp.float32),
        A_inv_T: wp.array3d(dtype=wp.float32),
        A_det: wp.array(dtype=wp.float32),
        c: wp.array2d(dtype=wp.float32),
        du: wp.array2d(dtype=wp.float32),
        local_as_shift: wp.array(dtype=wp.float32),
        opacity: wp.array(dtype=wp.float32),
        colors: wp.array(dtype=wp.float32),
        G_real: wp.array2d(dtype=wp.float32),
        G_imag: wp.array2d(dtype=wp.float32),
    ):
        pixel_y, pixel_x = wp.tid()
        zero = wp.float32(0.0)
        one = wp.float32(1.0)
        eps = wp.float32(1.0e-12)
        pi = wp.float32(_PI)
        two_pi = wp.float32(_TWO_PI)

        fx_val = fx[pixel_y, pixel_x]
        fy_val = fy[pixel_y, pixel_x]
        fz_val = fz[pixel_y, pixel_x]
        inv_wvl = one / wvl
        inv_wvl_sq = inv_wvl * inv_wvl

        real_accum = zero
        imag_accum = zero

        for gaussian_id in range(opacity.shape[0]):
            flx_val = (
                fx_val * R[gaussian_id, 0, 0]
                + fy_val * R[gaussian_id, 0, 1]
                + fz_val * R[gaussian_id, 0, 2]
            )
            fly_val = (
                fx_val * R[gaussian_id, 1, 0]
                + fy_val * R[gaussian_id, 1, 1]
                + fz_val * R[gaussian_id, 1, 2]
            )
            flz_sq = wp.max(inv_wvl_sq - flx_val * flx_val - fly_val * fly_val, eps)
            flz_val = wp.sqrt(flz_sq)

            fx_l_offset_val = flx_val - du[gaussian_id, 0]
            fy_l_offset_val = fly_val - du[gaussian_id, 1]

            fx_ref_val = (
                fx_l_offset_val * A_inv_T[gaussian_id, 0, 0]
                + fy_l_offset_val * A_inv_T[gaussian_id, 0, 1]
            )
            fy_ref_val = (
                fx_l_offset_val * A_inv_T[gaussian_id, 1, 0]
                + fy_l_offset_val * A_inv_T[gaussian_id, 1, 1]
            )

            gaussian_spectrum = two_pi * wp.exp(
                -wp.float32(2.0) * pi * pi * (fx_ref_val * fx_ref_val + fy_ref_val * fy_ref_val)
            )
            phase = two_pi * (
                flx_val * c[gaussian_id, 0]
                + fly_val * c[gaussian_id, 1]
                + flz_val * c[gaussian_id, 2]
            ) - two_pi * inv_wvl * local_as_shift[gaussian_id]
            scale = (flz_val / fz_val) * (one / A_det[gaussian_id])
            contribution = gaussian_spectrum * scale * opacity[gaussian_id] * colors[gaussian_id]

            real_accum = real_accum + contribution * wp.cos(phase)
            imag_accum = imag_accum + contribution * wp.sin(phase)

        G_real[pixel_y, pixel_x] = real_accum
        G_imag[pixel_y, pixel_x] = imag_accum


def warp_available() -> bool:
    return wp is not None


def warp_unavailable_reason() -> str:
    if _WARP_IMPORT_ERROR is None:
        return "Warp is not available."
    return f"Warp import failed: {_WARP_IMPORT_ERROR}"


def _ensure_warp_initialized() -> None:
    global _WARP_INITIALIZED
    if wp is None:
        raise RuntimeError(warp_unavailable_reason())
    if not _WARP_INITIALIZED:
        cache_dir = os.environ.get(
            "WARP_CACHE_DIR",
            os.path.join(tempfile.gettempdir(), "warp-cache"),
        )
        os.makedirs(cache_dir, exist_ok=True)
        wp.config.kernel_cache_dir = cache_dir
        wp.init()
        _WARP_INITIALIZED = True


def _warp_device_from_torch(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        return f"cuda:{index}"
    return "cpu"


def _validate_inputs(*tensors: torch.Tensor) -> None:
    dtypes = {tensor.dtype for tensor in tensors}
    devices = {tensor.device for tensor in tensors}
    if len(dtypes) != 1 or next(iter(dtypes)) != torch.float32:
        raise TypeError("Warp gaussian backend currently requires float32 tensors.")
    if len(devices) != 1:
        raise ValueError("Warp gaussian backend requires all tensors on the same device.")


def cgh_gaussians_naive(
    fx: torch.Tensor,
    fy: torch.Tensor,
    fz: torch.Tensor,
    wvl: float,
    R: torch.Tensor,
    A_inv_T: torch.Tensor,
    A_det: torch.Tensor,
    c: torch.Tensor,
    du: torch.Tensor,
    local_AS_shift: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ensure_warp_initialized()

    fx = fx.contiguous()
    fy = fy.contiguous()
    fz = fz.contiguous()
    R = R.contiguous()
    A_inv_T = A_inv_T.contiguous()
    A_det = A_det.contiguous()
    c = c.contiguous()
    du = du.contiguous()
    local_AS_shift = local_AS_shift.contiguous()
    opacity = opacity.contiguous()
    color = color.contiguous()
    _validate_inputs(fx, fy, fz, R, A_inv_T, A_det, c, du, local_AS_shift, opacity, color)

    G_real = torch.zeros_like(fx)
    G_imag = torch.zeros_like(fx)
    launch_kwargs = {
        "kernel": _cgh_gaussians_naive_kernel,
        "dim": fx.shape,
        "inputs": [fx, fy, fz, float(wvl), R, A_inv_T, A_det, c, du, local_AS_shift, opacity, color],
        "outputs": [G_real, G_imag],
        "device": _warp_device_from_torch(fx.device),
    }
    if fx.device.type == "cuda":
        launch_kwargs["stream"] = wp.stream_from_torch(
            torch.cuda.current_stream(device=fx.device)
        )

    wp.launch(**launch_kwargs)
    return G_real, G_imag
