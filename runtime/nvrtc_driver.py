"""NVRTC/CUDA Driver API wrappers."""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


def _candidate_paths(names: Iterable[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        found = ctypes.util.find_library(name)
        if found:
            out.append(found)
    return out


def _load_library(candidates: Sequence[str]) -> ctypes.CDLL:
    errors: list[str] = []
    is_windows = os.name == "nt"
    loader = ctypes.WinDLL if is_windows else ctypes.CDLL
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return loader(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise OSError("Could not load CUDA library. Tried:\n" + "\n".join(errors))


def _nvrtc_candidates() -> list[str]:
    if os.name == "nt":
        from_cuda_path: list[str] = []
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            from_cuda_path.extend(glob.glob(os.path.join(cuda_path, "bin", "nvrtc64_*.dll")))
        return from_cuda_path + ["nvrtc64_120_0.dll", "nvrtc64_121_0.dll", "nvrtc64_122_0.dll"]
    return ["libnvrtc.so", "libnvrtc.so.12", *_candidate_paths(["nvrtc"])]


def _cuda_driver_candidates() -> list[str]:
    if os.name == "nt":
        return ["nvcuda.dll"]
    return ["libcuda.so.1", "libcuda.so", *_candidate_paths(["cuda"])]


def _check_cuda(result: int, cuda_lib: ctypes.CDLL, api: str) -> None:
    if result == 0:
        return
    name_p = ctypes.c_char_p()
    desc_p = ctypes.c_char_p()
    if hasattr(cuda_lib, "cuGetErrorName"):
        cuda_lib.cuGetErrorName(result, ctypes.byref(name_p))
    if hasattr(cuda_lib, "cuGetErrorString"):
        cuda_lib.cuGetErrorString(result, ctypes.byref(desc_p))
    name = name_p.value.decode("utf-8") if name_p.value else "CUDA_ERROR_UNKNOWN"
    desc = desc_p.value.decode("utf-8") if desc_p.value else ""
    raise RuntimeError(f"{api} failed: {name} ({result}) {desc}".strip())


def _check_nvrtc(result: int, nvrtc_lib: ctypes.CDLL, api: str) -> None:
    if result == 0:
        return
    nvrtc_lib.nvrtcGetErrorString.restype = ctypes.c_char_p
    msg = nvrtc_lib.nvrtcGetErrorString(result)
    msg_text = msg.decode("utf-8") if msg else "NVRTC_ERROR_UNKNOWN"
    raise RuntimeError(f"{api} failed: {msg_text} ({result})")


@dataclass(frozen=True)
class LoadedKernel:
    module: ctypes.c_void_p
    function: ctypes.c_void_p


class CUDANVRTCDriver:
    """Thin wrapper for NVRTC compile and CUDA Driver kernel launch."""

    def __init__(self) -> None:
        self.cuda = _load_library(_cuda_driver_candidates())
        self.nvrtc = _load_library(_nvrtc_candidates())
        self._bind_cuda()
        self._bind_nvrtc()
        _check_cuda(self.cuda.cuInit(0), self.cuda, "cuInit")

    def _bind_cuda(self) -> None:
        self.cuda.cuInit.argtypes = [ctypes.c_uint]
        self.cuda.cuInit.restype = ctypes.c_int

        self.cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.cuda.cuDeviceGet.restype = ctypes.c_int

        self.cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.cuda.cuCtxGetCurrent.restype = ctypes.c_int

        self.cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self.cuda.cuCtxSetCurrent.restype = ctypes.c_int

        self.cuda.cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        self.cuda.cuDevicePrimaryCtxRetain.restype = ctypes.c_int

        self.cuda.cuModuleLoadDataEx.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.cuda.cuModuleLoadDataEx.restype = ctypes.c_int

        self.cuda.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self.cuda.cuModuleGetFunction.restype = ctypes.c_int

        self.cuda.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self.cuda.cuLaunchKernel.restype = ctypes.c_int

        self.cuda.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.cuda.cuGetErrorName.restype = ctypes.c_int
        self.cuda.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.cuda.cuGetErrorString.restype = ctypes.c_int

    def _bind_nvrtc(self) -> None:
        self.nvrtc.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.nvrtc.nvrtcCreateProgram.restype = ctypes.c_int

        self.nvrtc.nvrtcCompileProgram.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.nvrtc.nvrtcCompileProgram.restype = ctypes.c_int

        self.nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.nvrtc.nvrtcGetProgramLogSize.restype = ctypes.c_int
        self.nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.nvrtc.nvrtcGetProgramLog.restype = ctypes.c_int

        self.nvrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.nvrtc.nvrtcGetPTXSize.restype = ctypes.c_int
        self.nvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.nvrtc.nvrtcGetPTX.restype = ctypes.c_int

        self.nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.nvrtc.nvrtcDestroyProgram.restype = ctypes.c_int

        self.nvrtc.nvrtcGetErrorString.argtypes = [ctypes.c_int]
        self.nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p

    def ensure_context(self, device_index: int) -> None:
        torch.cuda._lazy_init()
        current = ctypes.c_void_p()
        _check_cuda(self.cuda.cuCtxGetCurrent(ctypes.byref(current)), self.cuda, "cuCtxGetCurrent")
        if current.value:
            return

        device = ctypes.c_int()
        _check_cuda(self.cuda.cuDeviceGet(ctypes.byref(device), int(device_index)), self.cuda, "cuDeviceGet")
        primary_ctx = ctypes.c_void_p()
        _check_cuda(
            self.cuda.cuDevicePrimaryCtxRetain(ctypes.byref(primary_ctx), device),
            self.cuda,
            "cuDevicePrimaryCtxRetain",
        )
        _check_cuda(self.cuda.cuCtxSetCurrent(primary_ctx), self.cuda, "cuCtxSetCurrent")

    def compile_to_ptx(self, source: str, device_index: int) -> bytes:
        major, minor = torch.cuda.get_device_capability(device_index)
        arch = f"--gpu-architecture=compute_{major}{minor}".encode("utf-8")
        options = [
            arch,
            b"--std=c++14",
            b"--use_fast_math",
            b"--fmad=true",
        ]
        option_array = (ctypes.c_char_p * len(options))(*options)

        program = ctypes.c_void_p()
        _check_nvrtc(
            self.nvrtc.nvrtcCreateProgram(
                ctypes.byref(program),
                source.encode("utf-8"),
                b"fused.cu",
                0,
                None,
                None,
            ),
            self.nvrtc,
            "nvrtcCreateProgram",
        )

        try:
            result = self.nvrtc.nvrtcCompileProgram(program, len(options), option_array)
            if result != 0:
                log = self._get_nvrtc_log(program)
                err = self.nvrtc.nvrtcGetErrorString(result).decode("utf-8")
                raise RuntimeError(f"NVRTC compile failed: {err}\n{log}")

            ptx_size = ctypes.c_size_t()
            _check_nvrtc(self.nvrtc.nvrtcGetPTXSize(program, ctypes.byref(ptx_size)), self.nvrtc, "nvrtcGetPTXSize")
            buffer = ctypes.create_string_buffer(ptx_size.value)
            _check_nvrtc(self.nvrtc.nvrtcGetPTX(program, buffer), self.nvrtc, "nvrtcGetPTX")
            return bytes(buffer.raw)
        finally:
            _check_nvrtc(self.nvrtc.nvrtcDestroyProgram(ctypes.byref(program)), self.nvrtc, "nvrtcDestroyProgram")

    def _get_nvrtc_log(self, program: ctypes.c_void_p) -> str:
        log_size = ctypes.c_size_t()
        _check_nvrtc(self.nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(log_size)), self.nvrtc, "nvrtcGetProgramLogSize")
        if log_size.value <= 1:
            return ""
        buf = ctypes.create_string_buffer(log_size.value)
        _check_nvrtc(self.nvrtc.nvrtcGetProgramLog(program, buf), self.nvrtc, "nvrtcGetProgramLog")
        return buf.value.decode("utf-8", errors="replace")

    def load_kernel(self, ptx: bytes, kernel_name: str) -> LoadedKernel:
        module = ctypes.c_void_p()
        ptx_buffer = ctypes.create_string_buffer(ptx)
        _check_cuda(
            self.cuda.cuModuleLoadDataEx(
                ctypes.byref(module),
                ctypes.cast(ptx_buffer, ctypes.c_void_p),
                0,
                None,
                None,
            ),
            self.cuda,
            "cuModuleLoadDataEx",
        )
        function = ctypes.c_void_p()
        _check_cuda(
            self.cuda.cuModuleGetFunction(ctypes.byref(function), module, kernel_name.encode("utf-8")),
            self.cuda,
            "cuModuleGetFunction",
        )
        return LoadedKernel(module=module, function=function)

    def launch(
        self,
        loaded_kernel: LoadedKernel,
        inputs: Sequence[torch.Tensor],
        output: torch.Tensor,
        block_size: int,
        grid_size: int,
        stream_handle: int,
    ) -> None:
        arg_values: list[ctypes._SimpleCData] = []
        for t in inputs:
            arg_values.append(ctypes.c_void_p(int(t.data_ptr())))
        arg_values.append(ctypes.c_void_p(int(output.data_ptr())))
        numel = ctypes.c_longlong(int(output.numel()))
        arg_values.append(numel)

        params = (ctypes.c_void_p * len(arg_values))()
        for i, arg in enumerate(arg_values):
            params[i] = ctypes.cast(ctypes.byref(arg), ctypes.c_void_p)

        _check_cuda(
            self.cuda.cuLaunchKernel(
                loaded_kernel.function,
                int(grid_size),
                1,
                1,
                int(block_size),
                1,
                1,
                0,
                ctypes.c_void_p(int(stream_handle)),
                params,
                None,
            ),
            self.cuda,
            "cuLaunchKernel",
        )


def is_nvrtc_available() -> bool:
    """Check whether NVRTC and CUDA driver libraries are loadable."""
    try:
        _load_library(_nvrtc_candidates())
        _load_library(_cuda_driver_candidates())
        return True
    except OSError:
        return False
