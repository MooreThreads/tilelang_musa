"""Annotation helpers exposed on the TileLang language surface."""

import threading
from typing import Callable

from tilelang import tvm
from tilelang.layout import Fragment, Layout
from tilelang.utils.language import is_fragment
from tvm.script.parser.tir import attr, block_attr, evaluate
from tvm.tir import FloatImm

__all__ = [
    "use_swizzle",
    "annotate_layout",
    "annotate_safe_value",
    "annotate_l2_hit_ratio",
    "annotate_restrict_buffers",
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


def annotate_layout(layout_map: dict, allow_reannotation: bool = False, allow_buffer_region: bool = False):
    """Annotate the layout of the buffer.

    Parameters
    ----------
    layout_map : dict
        Buffer-to-layout map.
    allow_reannotation : bool
        If False (default), keep original block-level semantics.
        If True, record an ordered manual-layout declaration that can update
        a buffer layout in later statements.
    allow_buffer_region : bool
        If False (default), reject BufferRegion keys.
        If True, allow BufferRegion keys and map them to their underlying buffer.
    """
    _layout_map = {}
    for buffer, layout in layout_map.items():
        if isinstance(buffer, tvm.tir.Buffer):
            if is_fragment(buffer):
                assert isinstance(layout, Fragment), f"for Fragment {buffer}, layout must be a Fragment, but got {type(layout)}"
            buffer_data = buffer.data
            target_shape = buffer.shape
        elif isinstance(buffer, tvm.tir.BufferRegion):
            if not allow_buffer_region:
                raise ValueError("BufferRegion is not allowed in annotate_layout unless allow_buffer_region=True")
            buffer_data = buffer.buffer.data
            target_shape = buffer.buffer.shape
        else:
            raise ValueError(f"Invalid annotate_layout key type: {type(buffer)}")
        if isinstance(layout, Layout):
            _layout_map[buffer_data] = layout
        elif isinstance(layout, Callable):
            _layout_map[buffer_data] = Layout(target_shape, layout)
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
        _l2_hit_ratio_map[buffer.data] = FloatImm("float32", float(hit_ratio))
    return block_attr({"l2_hit_ratio_map": _l2_hit_ratio_map})


def annotate_restrict_buffers(*buffers):
    """Mark the given buffer parameters as non-restrict.

    This annotation tells codegen to omit the `__restrict__` qualifier for the
    specified kernel buffer parameters. Use this when two (or more) buffers may
    alias, for example overlapping slices from the same base tensor.

    Example
    -------
    >>> @T.prim_func
    ... def buggy_kernel(x: T.Tensor((N,), T.float32),
    ...                  y: T.Tensor((N,), T.float32)):
    ...     T.annotate_restrict_buffers(x, y)
    ...     with T.Kernel(N, threads=32) as pid:
    ...         y[pid] = x[pid] + 1
    """
    if not buffers:
        return None
    data_vars = []
    for buf in buffers:
        try:
            data_vars.append(buf.data)
        except Exception as e:
            raise TypeError(f"annotate_restrict_buffers expects Buffer arguments, got {type(buf)}") from e
    # Also return as block attribute (root block exists by default) for readability/tools.
    return block_attr({"tl.non_restrict_params": data_vars})
