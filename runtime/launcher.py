"""Kernel launch orchestration."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Sequence

import torch

from .cache import KernelCache
from .nvrtc_driver import CUDANVRTCDriver, LoadedKernel


@dataclass(frozen=True)
class LaunchRequest:
    cache_key: str
    source: str
    kernel_name: str


class KernelLauncher:
    """Compile (if needed), cache, and launch fused CUDA kernels."""

    def __init__(self, cache: KernelCache | None = None) -> None:
        self.cache = cache or KernelCache()
        self.driver = CUDANVRTCDriver()
        self._loaded: Dict[str, LoadedKernel] = {}
        self._lock = threading.Lock()

    def _ensure_loaded(self, request: LaunchRequest, device_index: int) -> LoadedKernel:
        with self._lock:
            if request.cache_key in self._loaded:
                return self._loaded[request.cache_key]

            artifact = self.cache.get(request.cache_key)
            if artifact is None:
                ptx = self.driver.compile_to_ptx(request.source, device_index=device_index)
                artifact = self.cache.put(request.cache_key, ptx)

            loaded = self.driver.load_kernel(artifact.ptx, request.kernel_name)
            self._loaded[request.cache_key] = loaded
            return loaded

    def launch(self, request: LaunchRequest, inputs: Sequence[torch.Tensor], output: torch.Tensor) -> None:
        if output.device.type != "cuda":
            raise ValueError("Output tensor must be on CUDA device")
        if not inputs:
            raise ValueError("inputs must be non-empty")

        device_index = int(output.device.index or 0)
        self.driver.ensure_context(device_index=device_index)
        loaded = self._ensure_loaded(request=request, device_index=device_index)

        block_size = 256
        grid_size = min(max((int(output.numel()) + block_size - 1) // block_size, 1), 65535)
        stream_handle = int(torch.cuda.current_stream(output.device).cuda_stream)
        self.driver.launch(
            loaded_kernel=loaded,
            inputs=inputs,
            output=output,
            block_size=block_size,
            grid_size=grid_size,
            stream_handle=stream_handle,
        )


_GLOBAL_LAUNCHER: KernelLauncher | None = None
_GLOBAL_LOCK = threading.Lock()


def get_global_launcher() -> KernelLauncher:
    global _GLOBAL_LAUNCHER
    with _GLOBAL_LOCK:
        if _GLOBAL_LAUNCHER is None:
            _GLOBAL_LAUNCHER = KernelLauncher()
        return _GLOBAL_LAUNCHER
