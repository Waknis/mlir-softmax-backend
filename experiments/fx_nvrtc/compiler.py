"""High-level compiler API."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.fx
import torch.nn as nn

from .codegen_cuda import generate_cuda_source
from .fx_parser import ParsedGraph, UnsupportedGraphError, parse_fx_graph
from .launcher import LaunchRequest, get_global_launcher


def _resolve_attr(root: nn.Module, qualified_name: str) -> Any:
    obj: Any = root
    for part in qualified_name.split("."):
        obj = getattr(obj, part)
    return obj


def _broadcast_strides(input_shape: Sequence[int], output_shape: Sequence[int]) -> tuple[int, ...]:
    rank_out = len(output_shape)
    rank_in = len(input_shape)
    if rank_in > rank_out:
        raise ValueError(f"Input rank {rank_in} cannot broadcast to output rank {rank_out}")

    contig_in: list[int] = []
    running = 1
    for dim in reversed(input_shape):
        contig_in.append(running)
        running *= int(dim)
    contig_in = list(reversed(contig_in))

    pad = rank_out - rank_in
    aligned_shape = (1,) * pad + tuple(int(d) for d in input_shape)
    aligned_strides = (0,) * pad + tuple(contig_in)

    out: list[int] = []
    for in_dim, out_dim, stride in zip(aligned_shape, output_shape, aligned_strides):
        if in_dim == out_dim:
            out.append(int(stride))
        elif in_dim == 1:
            out.append(0)
        else:
            raise ValueError(f"Cannot broadcast dim {in_dim} to {out_dim}")
    return tuple(out)


def _make_cache_key(
    graph_hash: str,
    shapes: Sequence[Sequence[int]],
    dtype: torch.dtype,
    device: torch.device,
) -> str:
    device_index = int(device.index or 0)
    raw = repr(
        (
            graph_hash,
            tuple(tuple(int(x) for x in shape) for shape in shapes),
            str(dtype),
            f"{device.type}:{device_index}",
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class CompiledModule:
    """Compiled module wrapper.

    This MVP currently targets inference-only elementwise CUDA graphs.
    Unsupported graphs or unsupported runtimes fall back to eager.
    """

    module: nn.Module
    _gm: torch.fx.GraphModule | None = field(default=None, init=False)
    _parsed: ParsedGraph | None = field(default=None, init=False)
    _signature: inspect.Signature = field(init=False)
    _enabled: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        self._signature = inspect.signature(self.module.forward)

    def _ensure_parsed(self) -> None:
        if self._gm is not None and self._parsed is not None:
            return
        try:
            self._gm = torch.fx.symbolic_trace(self.module)
            self._parsed = parse_fx_graph(self._gm)
        except UnsupportedGraphError:
            self._enabled = False
            self._gm = None
            self._parsed = None
        except Exception:
            self._enabled = False
            self._gm = None
            self._parsed = None

    def _collect_inputs(self, *args: Any, **kwargs: Any) -> list[torch.Tensor]:
        if self._parsed is None:
            raise RuntimeError("Parsed graph is missing")
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()

        tensors: list[torch.Tensor] = []
        for src in self._parsed.input_sources:
            if src.kind == "arg":
                if src.name not in bound.arguments:
                    raise ValueError(f"Missing argument for placeholder {src.name}")
                value = bound.arguments[src.name]
            elif src.kind == "attr":
                value = _resolve_attr(self.module, src.name)
            else:
                raise ValueError(f"Unknown input source kind: {src.kind}")

            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Only tensor inputs are supported, got {type(value)!r} from {src}")
            tensors.append(value)
        return tensors

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not self._enabled:
            return self.module(*args, **kwargs)
        if not torch.cuda.is_available():
            return self.module(*args, **kwargs)

        self._ensure_parsed()
        if not self._enabled or self._parsed is None:
            return self.module(*args, **kwargs)

        try:
            raw_inputs = self._collect_inputs(*args, **kwargs)
        except Exception:
            self._enabled = False
            return self.module(*args, **kwargs)

        if not raw_inputs:
            return self.module(*args, **kwargs)

        ref_device = raw_inputs[0].device
        ref_dtype = raw_inputs[0].dtype

        if ref_device.type != "cuda":
            return self.module(*args, **kwargs)
        if ref_dtype not in {torch.float16, torch.float32}:
            return self.module(*args, **kwargs)
        if any(inp.device != ref_device for inp in raw_inputs):
            return self.module(*args, **kwargs)
        if any(inp.dtype != ref_dtype for inp in raw_inputs):
            return self.module(*args, **kwargs)
        if any(inp.requires_grad for inp in raw_inputs):
            return self.module(*args, **kwargs)

        input_shapes = [tuple(int(d) for d in inp.shape) for inp in raw_inputs]
        try:
            output_shape = tuple(int(d) for d in torch.broadcast_shapes(*input_shapes))
        except RuntimeError:
            return self.module(*args, **kwargs)

        input_broadcast_strides: list[tuple[int, ...]] = []
        try:
            for shape in input_shapes:
                input_broadcast_strides.append(_broadcast_strides(shape, output_shape))
        except ValueError:
            return self.module(*args, **kwargs)

        contiguous_inputs = [inp.contiguous() for inp in raw_inputs]
        out = torch.empty(output_shape, device=ref_device, dtype=ref_dtype)
        if out.numel() == 0:
            return out

        cache_key = _make_cache_key(
            graph_hash=self._parsed.graph_hash,
            shapes=input_shapes,
            dtype=ref_dtype,
            device=ref_device,
        )
        dtype_name = "float16" if ref_dtype == torch.float16 else "float32"
        spec = generate_cuda_source(
            expr=self._parsed.expr,
            input_count=len(contiguous_inputs),
            dtype=dtype_name,
            graph_hash=self._parsed.graph_hash,
            output_shape=output_shape,
            input_broadcast_strides=input_broadcast_strides,
        )

        launcher = get_global_launcher()
        request = LaunchRequest(
            cache_key=cache_key,
            source=spec.source,
            kernel_name=spec.kernel_name,
        )

        try:
            launcher.launch(request=request, inputs=contiguous_inputs, output=out)
            return out
        except Exception:
            self._enabled = False
            return self.module(*args, **kwargs)


def compile_module(module: nn.Module) -> CompiledModule:
    """Compile an nn.Module to a fused runtime callable."""
    return CompiledModule(module=module.eval())
