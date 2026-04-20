"""Runtime cache for compiled kernel artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class KernelArtifact:
    key: str
    ptx: bytes


class KernelCache:
    """Memory + disk cache for compiled kernels."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._mem: Dict[str, KernelArtifact] = {}
        self.cache_dir = cache_dir or Path.home() / ".cache" / "mlir_softmax_backend" / "fx_nvrtc"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[KernelArtifact]:
        if key in self._mem:
            return self._mem[key]
        file_path = self.cache_dir / f"{key}.ptx"
        if not file_path.exists():
            return None
        artifact = KernelArtifact(key=key, ptx=file_path.read_bytes())
        self._mem[key] = artifact
        return artifact

    def put(self, key: str, ptx: bytes) -> KernelArtifact:
        artifact = KernelArtifact(key=key, ptx=ptx)
        self._mem[key] = artifact
        (self.cache_dir / f"{key}.ptx").write_bytes(ptx)
        return artifact
