"""ctypes loader for the hand-written softmax CUDA kernels.

Loads ``libmlc_softmax_kernels.so`` (built by ``kernels/CMakeLists.txt`` or
directly via ``nvcc``) and exposes callables that accept CUDA-device tensors
and run the kernel on the current stream. Tensors stay on the GPU throughout
-- no host round-trips.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import torch


_REPO_ROOT = Path(__file__).resolve().parent.parent
_LIB_NAME = "libmlc_softmax_kernels.so"

_SEARCH_PATHS = (
    _REPO_ROOT / "build" / "lib" / _LIB_NAME,
    _REPO_ROOT / "kernels" / _LIB_NAME,
    _REPO_ROOT / _LIB_NAME,
)


class SoftmaxKernelUnavailable(RuntimeError):
    pass


def _find_library() -> Path:
    override = os.environ.get("MLC_SOFTMAX_KERNELS_LIB")
    if override:
        path = Path(override)
        if path.exists():
            return path
    for candidate in _SEARCH_PATHS:
        if candidate.exists():
            return candidate
    raise SoftmaxKernelUnavailable(
        f"Could not locate {_LIB_NAME}. Searched: "
        + ", ".join(str(p) for p in _SEARCH_PATHS)
        + ". Set MLC_SOFTMAX_KERNELS_LIB to override, or run "
        "`cmake --build build --target mlc_softmax_kernels`."
    )


class _KernelLibrary:
    def __init__(self) -> None:
        self._lib = ctypes.CDLL(str(_find_library()))

        for name in ("mlc_softmax_online_f32", "mlc_softmax_naive_f32"):
            fn = getattr(self._lib, name)
            fn.argtypes = [
                ctypes.c_void_p,   # d_input (float*)
                ctypes.c_void_p,   # d_output (float*)
                ctypes.c_int64,    # rows
                ctypes.c_int64,    # cols
                ctypes.c_void_p,   # cuda_stream
            ]
            fn.restype = ctypes.c_int

        fn = self._lib.mlc_softmax_online_f16
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int

        self._lib.mlc_softmax_online_last_error.argtypes = [ctypes.c_int]
        self._lib.mlc_softmax_online_last_error.restype = ctypes.c_char_p

    def _check(self, code: int) -> None:
        if code != 0:
            msg = self._lib.mlc_softmax_online_last_error(code)
            raise RuntimeError(
                f"CUDA kernel launch failed (code={code}): "
                f"{msg.decode() if msg else 'unknown'}"
            )

    def online_f32(self, x: torch.Tensor, y: torch.Tensor) -> None:
        _validate(x, y, torch.float32)
        stream = torch.cuda.current_stream(x.device).cuda_stream
        rc = self._lib.mlc_softmax_online_f32(
            x.data_ptr(), y.data_ptr(),
            x.shape[0], x.shape[1],
            stream,
        )
        self._check(rc)

    def naive_f32(self, x: torch.Tensor, y: torch.Tensor) -> None:
        _validate(x, y, torch.float32)
        stream = torch.cuda.current_stream(x.device).cuda_stream
        rc = self._lib.mlc_softmax_naive_f32(
            x.data_ptr(), y.data_ptr(),
            x.shape[0], x.shape[1],
            stream,
        )
        self._check(rc)

    def online_f16(self, x: torch.Tensor, y: torch.Tensor) -> None:
        _validate(x, y, torch.float16)
        stream = torch.cuda.current_stream(x.device).cuda_stream
        rc = self._lib.mlc_softmax_online_f16(
            x.data_ptr(), y.data_ptr(),
            x.shape[0], x.shape[1],
            stream,
        )
        self._check(rc)


def _validate(x: torch.Tensor, y: torch.Tensor, dtype: torch.dtype) -> None:
    if x.device.type != "cuda" or y.device.type != "cuda":
        raise ValueError("Both tensors must be CUDA tensors")
    if x.dtype != dtype or y.dtype != dtype:
        raise ValueError(f"Both tensors must be dtype {dtype}, got {x.dtype}/{y.dtype}")
    if x.shape != y.shape or x.dim() != 2:
        raise ValueError(f"Shapes must match and be 2-D, got {tuple(x.shape)} / {tuple(y.shape)}")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("Tensors must be contiguous (row-major)")


_CACHED: Optional[_KernelLibrary] = None


def get_library() -> _KernelLibrary:
    global _CACHED
    if _CACHED is None:
        _CACHED = _KernelLibrary()
    return _CACHED
