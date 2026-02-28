"""Annotation helpers exposed on the TileLang language surface."""
from __future__ import annotations

from typing import Callable
import threading

from tilelang import tvm
from tilelang.layout import Layout
from tvm.script.parser.tir import attr, block_attr, evaluate

__all__ = [
    "use_swizzle",
    "annotate_layout",
    "annotate_safe_value",
    "annotate_l2_hit_ratio",
]

_tls = threading.local()


def _next_layout_override_step() -> int:
    if not hasattr(_tls, "layout_override_step"):
        _tls.layout_override_step = 0
    step = _tls.layout_override_step
    _tls.layout_override_step += 1
    return step


def use_swizzle(panel_size: int, order: str = "row", enable: bool = True):
    """Annotate a kernel to use a specific threadblock swizzle pattern."""
    device_func = "rasterization2DRow" if order == "row" else "rasterization2DColumn"
    if not enable:
        return None
    return attr(None, "threadblock_swizzle_pattern", f"tl::{device_func}<{panel_size}>")


def annotate_layout(layout_map: dict, allow_reannotation: bool = False):
    """Annotate the layout of the buffer.

    Parameters
    ----------
    layout_map : dict
        Buffer-to-layout map.
    allow_reannotation : bool
        If False (default), keep original block-level semantics.
        If True, record an ordered manual-layout declaration that can update
        a buffer layout in later statements.
    """
    _layout_map = {}
    for buffer, layout in layout_map.items():
        if isinstance(layout, Layout):
            _layout_map[buffer.data] = layout
        elif isinstance(layout, Callable):
            _layout_map[buffer.data] = Layout(buffer.shape, layout)
        else:
            raise ValueError(f"Invalid layout: {layout}")

    if not allow_reannotation:
        return block_attr({"layout_map": _layout_map})

    step = _next_layout_override_step()
    block_attr({"layout_override_seq": {str(step): _layout_map}})
    marker_op = tvm.ir.Op.get("tl.layout_marker")
    evaluate(tvm.tir.Call("int32", marker_op, [tvm.tir.IntImm("int32", step)]))
    return None


def annotate_safe_value(safe_value_map: dict):
    """Annotate the safe value of the buffer."""
    _safe_value_map = {}
    for buffer, safe_value in safe_value_map.items():
        _safe_value_map[buffer.data] = safe_value
    return block_attr({"safe_value_map": _safe_value_map})


def annotate_l2_hit_ratio(l2_hit_ratio_map: dict):
    """Annotate the L2 hit ratio of the buffer."""
    _l2_hit_ratio_map = {}
    for buffer, hit_ratio in l2_hit_ratio_map.items():
        assert buffer.scope() == "global", "persistent L2 can only be applied to global buffers"
        _l2_hit_ratio_map[buffer.data] = float(hit_ratio)
    return block_attr({"l2_hit_ratio_map": _l2_hit_ratio_map})
