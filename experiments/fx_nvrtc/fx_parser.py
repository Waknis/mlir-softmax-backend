"""FX graph parser for supported fused elementwise chains."""

from __future__ import annotations

import hashlib
import operator
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F

from .ir import BinaryExpr, ConstExpr, Expr, InputExpr, UnaryExpr


class UnsupportedGraphError(RuntimeError):
    """Raised when the FX graph contains unsupported nodes for MVP fusion."""


@dataclass(frozen=True)
class InputSource:
    kind: str  # "arg" or "attr"
    name: str


@dataclass(frozen=True)
class ParsedGraph:
    expr: Expr
    input_sources: Tuple[InputSource, ...]
    graph_hash: str


_BINARY_FUNCTIONS = {
    operator.add: "add",
    torch.add: "add",
    operator.mul: "mul",
    torch.mul: "mul",
    operator.sub: "sub",
    torch.sub: "sub",
    operator.truediv: "div",
    torch.div: "div",
}

_UNARY_FUNCTIONS = {
    torch.exp: "exp",
    torch.tanh: "tanh",
    torch.sigmoid: "sigmoid",
    torch.relu: "relu",
    F.gelu: "gelu",
    torch.nn.functional.gelu: "gelu",
}


def _normalize_binary_from_target(target: object) -> str | None:
    if target in _BINARY_FUNCTIONS:
        return _BINARY_FUNCTIONS[target]
    target_repr = str(target)
    if "aten.add" in target_repr:
        return "add"
    if "aten.mul" in target_repr:
        return "mul"
    if "aten.sub" in target_repr:
        return "sub"
    if "aten.div" in target_repr:
        return "div"
    return None


def _normalize_unary_from_target(target: object) -> str | None:
    if target in _UNARY_FUNCTIONS:
        return _UNARY_FUNCTIONS[target]
    target_repr = str(target)
    if "aten.exp" in target_repr:
        return "exp"
    if "aten.tanh" in target_repr:
        return "tanh"
    if "aten.sigmoid" in target_repr:
        return "sigmoid"
    if "aten.relu" in target_repr:
        return "relu"
    if "aten.gelu" in target_repr:
        return "gelu"
    return None


def _parse_arg(
    value: object,
    expr_map: Dict[torch.fx.Node, Expr],
) -> Expr:
    if isinstance(value, torch.fx.Node):
        if value not in expr_map:
            raise UnsupportedGraphError(f"Node used before defined: {value}")
        return expr_map[value]
    if isinstance(value, (float, int)):
        return ConstExpr(float(value))
    raise UnsupportedGraphError(f"Unsupported argument type: {type(value)!r}")


def _collect_input_indices(expr: Expr) -> set[int]:
    if isinstance(expr, InputExpr):
        return {expr.index}
    if isinstance(expr, ConstExpr):
        return set()
    if isinstance(expr, UnaryExpr):
        return _collect_input_indices(expr.x)
    if isinstance(expr, BinaryExpr):
        return _collect_input_indices(expr.lhs) | _collect_input_indices(expr.rhs)
    raise TypeError(f"Unexpected expr type: {type(expr)!r}")


def _remap_expr_inputs(expr: Expr, index_map: Dict[int, int]) -> Expr:
    if isinstance(expr, InputExpr):
        return InputExpr(index=index_map[expr.index])
    if isinstance(expr, ConstExpr):
        return expr
    if isinstance(expr, UnaryExpr):
        return UnaryExpr(op=expr.op, x=_remap_expr_inputs(expr.x, index_map))
    if isinstance(expr, BinaryExpr):
        return BinaryExpr(
            op=expr.op,
            lhs=_remap_expr_inputs(expr.lhs, index_map),
            rhs=_remap_expr_inputs(expr.rhs, index_map),
        )
    raise TypeError(f"Unexpected expr type: {type(expr)!r}")


def parse_fx_graph(gm: torch.fx.GraphModule) -> ParsedGraph:
    """Parse FX graph into a single fused elementwise expression."""
    expr_map: Dict[torch.fx.Node, Expr] = {}
    input_sources: List[InputSource] = []
    output_expr: Expr | None = None

    named_modules = dict(gm.named_modules())

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            input_index = len(input_sources)
            expr_map[node] = InputExpr(input_index)
            input_sources.append(InputSource(kind="arg", name=str(node.target)))
            continue

        if node.op == "get_attr":
            input_index = len(input_sources)
            expr_map[node] = InputExpr(input_index)
            input_sources.append(InputSource(kind="attr", name=str(node.target)))
            continue

        if node.op == "call_function":
            binary_name = _normalize_binary_from_target(node.target)
            unary_name = _normalize_unary_from_target(node.target)

            if binary_name is not None:
                if len(node.args) != 2:
                    raise UnsupportedGraphError(
                        f"Binary op expects 2 args, got {len(node.args)}: {node.target}"
                    )
                if "alpha" in node.kwargs and float(node.kwargs["alpha"]) != 1.0:
                    raise UnsupportedGraphError("torch.add/sub alpha != 1.0 is not supported")
                lhs = _parse_arg(node.args[0], expr_map)
                rhs = _parse_arg(node.args[1], expr_map)
                expr_map[node] = BinaryExpr(op=binary_name, lhs=lhs, rhs=rhs)
                continue

            if unary_name is not None:
                if len(node.args) != 1:
                    raise UnsupportedGraphError(
                        f"Unary op expects 1 arg, got {len(node.args)}: {node.target}"
                    )
                x = _parse_arg(node.args[0], expr_map)
                expr_map[node] = UnaryExpr(op=unary_name, x=x)
                continue

            raise UnsupportedGraphError(f"Unsupported call_function target: {node.target}")

        if node.op == "call_method":
            method = str(node.target)
            if method in {"add", "__add__", "mul", "__mul__", "sub", "__sub__", "div", "__truediv__"}:
                if len(node.args) < 2:
                    raise UnsupportedGraphError(f"Method {method} expects lhs/rhs args")
                lhs = _parse_arg(node.args[0], expr_map)
                rhs = _parse_arg(node.args[1], expr_map)
                op_map = {
                    "add": "add",
                    "__add__": "add",
                    "mul": "mul",
                    "__mul__": "mul",
                    "sub": "sub",
                    "__sub__": "sub",
                    "div": "div",
                    "__truediv__": "div",
                }
                expr_map[node] = BinaryExpr(op=op_map[method], lhs=lhs, rhs=rhs)
                continue
            if method in {"exp", "tanh", "sigmoid", "relu", "gelu"}:
                if len(node.args) < 1:
                    raise UnsupportedGraphError(f"Method {method} expects one arg")
                x = _parse_arg(node.args[0], expr_map)
                expr_map[node] = UnaryExpr(op=method, x=x)
                continue
            raise UnsupportedGraphError(f"Unsupported call_method target: {method}")

        if node.op == "call_module":
            submod = named_modules.get(str(node.target))
            if submod is None:
                raise UnsupportedGraphError(f"Missing submodule for target: {node.target}")
            unary_map: Dict[type[nn.Module], str] = {
                nn.ReLU: "relu",
                nn.GELU: "gelu",
                nn.Tanh: "tanh",
                nn.Sigmoid: "sigmoid",
            }
            op_name = unary_map.get(type(submod))
            if op_name is None:
                raise UnsupportedGraphError(
                    f"Unsupported call_module {type(submod).__name__} at {node.target}"
                )
            if len(node.args) != 1:
                raise UnsupportedGraphError(
                    f"Unary call_module expects 1 arg, got {len(node.args)}"
                )
            x = _parse_arg(node.args[0], expr_map)
            expr_map[node] = UnaryExpr(op=op_name, x=x)
            continue

        if node.op == "output":
            if len(node.args) != 1:
                raise UnsupportedGraphError("Only single-output graphs are supported")
            out_arg = node.args[0]
            if isinstance(out_arg, (tuple, list)):
                raise UnsupportedGraphError("Tuple/list outputs are not supported for MVP")
            output_expr = _parse_arg(out_arg, expr_map)
            continue

        raise UnsupportedGraphError(f"Unsupported node type: {node.op}")

    if output_expr is None:
        raise UnsupportedGraphError("FX graph has no output expression")

    # Prune placeholders/get_attrs that are not used in the final fused expression.
    used = sorted(_collect_input_indices(output_expr))
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used)}
    remapped_expr = _remap_expr_inputs(output_expr, index_map=index_map)
    remapped_sources = tuple(input_sources[idx] for idx in used)

    stable = remapped_expr.stable_repr() + "|" + ",".join(
        f"{src.kind}:{src.name}" for src in remapped_sources
    )
    graph_hash = hashlib.sha256(stable.encode("utf-8")).hexdigest()
    return ParsedGraph(
        expr=remapped_expr,
        input_sources=remapped_sources,
        graph_hash=graph_hash,
    )
