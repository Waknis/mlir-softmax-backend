"""Expression IR nodes used for fused elementwise codegen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias, Union


@dataclass(frozen=True)
class InputExpr:
    index: int

    def stable_repr(self) -> str:
        return f"input({self.index})"


@dataclass(frozen=True)
class ConstExpr:
    value: float

    def stable_repr(self) -> str:
        return f"const({self.value:.8f})"


@dataclass(frozen=True)
class UnaryExpr:
    op: str
    x: "Expr"

    def stable_repr(self) -> str:
        return f"{self.op}({self.x.stable_repr()})"


@dataclass(frozen=True)
class BinaryExpr:
    op: str
    lhs: "Expr"
    rhs: "Expr"

    def stable_repr(self) -> str:
        return f"{self.op}({self.lhs.stable_repr()},{self.rhs.stable_repr()})"


Expr: TypeAlias = Union[InputExpr, ConstExpr, UnaryExpr, BinaryExpr]


def emit_c_float_expr(expr: Expr, input_name: Callable[[int], str]) -> str:
    """Emit CUDA C expression in float domain.

    `input_name(i)` should return a float-valued identifier/expression.
    """
    if isinstance(expr, InputExpr):
        return input_name(expr.index)
    if isinstance(expr, ConstExpr):
        return f"{expr.value:.8f}f"
    if isinstance(expr, BinaryExpr):
        lhs = emit_c_float_expr(expr.lhs, input_name)
        rhs = emit_c_float_expr(expr.rhs, input_name)
        if expr.op == "add":
            return f"({lhs} + {rhs})"
        if expr.op == "sub":
            return f"({lhs} - {rhs})"
        if expr.op == "mul":
            return f"({lhs} * {rhs})"
        if expr.op == "div":
            return f"({lhs} / {rhs})"
        raise ValueError(f"Unsupported binary op: {expr.op}")
    if isinstance(expr, UnaryExpr):
        x = emit_c_float_expr(expr.x, input_name)
        if expr.op == "exp":
            return f"__expf({x})"
        if expr.op == "tanh":
            return f"tanhf({x})"
        if expr.op == "sigmoid":
            return f"(1.0f / (1.0f + __expf(-({x}))))"
        if expr.op == "relu":
            return f"fmaxf({x}, 0.0f)"
        if expr.op == "gelu":
            return (
                "(0.5f * ({x}) * (1.0f + tanhf(0.79788456f * "
                "(({x}) + 0.044715f * ({x}) * ({x}) * ({x})))))"
            ).format(x=x)
        raise ValueError(f"Unsupported unary op: {expr.op}")
    raise TypeError(f"Unexpected expr type: {type(expr)!r}")
