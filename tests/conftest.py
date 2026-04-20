from __future__ import annotations

import pytest
import torch

from experiments.fx_nvrtc.nvrtc_driver import is_nvrtc_available


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def cuda_nvrtc_available(cuda_available: bool) -> bool:
    return cuda_available and is_nvrtc_available()
