from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx

from experiments.fx_nvrtc.codegen_cuda import generate_cuda_source
from experiments.fx_nvrtc.compiler import _broadcast_strides
from experiments.fx_nvrtc.fx_parser import parse_fx_graph


class GeluPattern(nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.gelu(x * w + b)


class ReluModulePattern(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.relu(x + y)


class UnusedInputPattern(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del y
        return torch.relu(x)


def test_fx_parser_gelu_chain() -> None:
    gm = torch.fx.symbolic_trace(GeluPattern())
    parsed = parse_fx_graph(gm)
    assert len(parsed.input_sources) == 3
    assert parsed.expr.stable_repr().startswith("gelu(")


def test_fx_parser_call_module_relu() -> None:
    gm = torch.fx.symbolic_trace(ReluModulePattern())
    parsed = parse_fx_graph(gm)
    assert len(parsed.input_sources) == 2
    assert parsed.expr.stable_repr().startswith("relu(")


def test_fx_parser_prunes_unused_inputs() -> None:
    gm = torch.fx.symbolic_trace(UnusedInputPattern())
    parsed = parse_fx_graph(gm)
    assert len(parsed.input_sources) == 1


def test_broadcast_stride_helper() -> None:
    assert _broadcast_strides((8, 16), (8, 16)) == (16, 1)
    assert _broadcast_strides((16,), (8, 16)) == (0, 1)
    assert _broadcast_strides((1, 16), (8, 16)) == (0, 1)


def test_broadcast_stride_helper_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="cannot broadcast"):
        _broadcast_strides((2, 16, 4), (16, 4))

    with pytest.raises(ValueError, match="Cannot broadcast"):
        _broadcast_strides((3, 16), (8, 16))


def test_codegen_emits_broadcast_indexing_when_needed() -> None:
    gm = torch.fx.symbolic_trace(ReluModulePattern())
    parsed = parse_fx_graph(gm)
    spec = generate_cuda_source(
        expr=parsed.expr,
        input_count=2,
        dtype="float32",
        graph_hash=parsed.graph_hash,
        output_shape=(8, 16),
        input_broadcast_strides=((16, 1), (0, 1)),
    )
    assert spec.has_broadcast
    assert spec.vector_width == 1
    assert "in_offset_1" in spec.source
