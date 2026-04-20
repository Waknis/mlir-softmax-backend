"""Fusion utilities for elementwise chains."""

from __future__ import annotations


def is_reduction_op(op_name: str) -> bool:
    """Simple reducer guard used by fusion logic."""
    return op_name in {"sum", "mean", "amax", "amin"}
