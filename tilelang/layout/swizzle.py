"""Wrapping Layouts."""

# pylint: disable=invalid-name, unsupported-binary-operation
from __future__ import annotations

import tvm
from tvm import tir
from tilelang import _ffi_api
from tilelang._typing import BufferLikeType, BufferLikeTypeTuple
from .layout import Layout


def _get_buffer_info(buffer_or_load_or_region: BufferLikeType) -> tuple[tir.Buffer, list[int], str]:
    """
    Extract buffer, shape, and dtype from BufferLikeType.

    Args:
        buffer_or_load_or_region: BufferLikeType

    Returns:
        tuple: (buffer, shape, dtype)
    """
    if isinstance(buffer_or_load_or_region, tir.Buffer):
        return buffer_or_load_or_region, buffer_or_load_or_region.shape, buffer_or_load_or_region.dtype
    elif isinstance(buffer_or_load_or_region, BufferLikeTypeTuple):
        buf = buffer_or_load_or_region.buffer
        return buf, buf.shape, buf.dtype
    else:
        raise TypeError(f"Expected BufferLikeType, got {type(buffer_or_load_or_region)}")


def _get_stride_continuous(buffer_or_load_or_region: BufferLikeType) -> tuple[int, int]:
    """
    Get stride (product of all dims except the last) and continuous (last dimension)
    from BufferLikeType.

    Args:
        buffer_or_load_or_region: BufferLikeType

    Returns:
        tuple: (stride, continuous) as integers
    """
    _, shape, _ = _get_buffer_info(buffer_or_load_or_region)
    stride = 1
    for dim in shape[:-1]:
        stride *= int(dim)
    continuous = int(shape[-1])
    return stride, continuous


def _get_element_size(buffer_or_load_or_region: BufferLikeType) -> int:
    """
    Get element size in bits from BufferLikeType.

    Args:
        buffer_or_load_or_region: BufferLikeType

    Returns:
        int: Element size in bits
    """
    _, _, dtype = _get_buffer_info(buffer_or_load_or_region)
    return int(tvm.DataType(dtype).bits)


# Use a stable swizzled layout to ensure consistent memory access patterns.
# Swizzling should be enabled or disabled based on whether TMA (Tensor Memory Access) is applied.
def make_swizzled_layout(buffer: BufferLikeType, k_major: bool = True, allow_pad: bool = True):
    _, shape, _ = _get_buffer_info(buffer)
    stride, continuous = _get_stride_continuous(buffer)
    element_size = _get_element_size(buffer)
    base = _ffi_api.make_swizzled_layout(
        stride,
        continuous,
        element_size,
        k_major,
        allow_pad,
    )
    return base.reshape(shape)


# for Volta Intrinsics
def make_volta_swizzled_layout(buffer: BufferLikeType, is_a: bool = True, k_inner: bool = True):
    _, shape, _ = _get_buffer_info(buffer)
    stride, continuous = _get_stride_continuous(buffer)
    base = _ffi_api.make_volta_swizzled_layout(
        stride,
        continuous,
        is_a,
        k_inner,
    )
    return base.reshape(shape)


# for WGMMA Intrinsics
def make_wgmma_swizzled_layout(buffer: BufferLikeType, continuity: int = None, k_major: bool = True):
    _, shape, _ = _get_buffer_info(buffer)
    stride, continuous = _get_stride_continuous(buffer)
    element_size = _get_element_size(buffer)
    if continuity is None:
        continuity = continuous
    base = _ffi_api.make_wgmma_swizzled_layout(
        stride,
        continuous,
        continuity,
        element_size,
        k_major,
    )
    return base.reshape(shape)


# for SQMMA Intrinsics (PH1)
def make_sqmma_swizzled_layout(buffer: BufferLikeType, continuity: int = None, k_major: bool = True):
    if isinstance(buffer, tir.Buffer):
        shape = list(buffer.shape)
        stride = int(shape[0])
        continuous = int(shape[1])
        if continuity is None:
            continuity = continuous
        base = _ffi_api.make_sqmma_swizzled_layout(
            stride,
            continuous,
            continuity,
            int(tvm.DataType(buffer.dtype).bits),
            k_major,
        )
        return base.reshape(shape)

    if isinstance(buffer, tir.BufferRegion):
        region_shape = [r.extent for r in buffer.region]
        if len(region_shape) < 2:
            raise ValueError(f"make_sqmma_swizzled_layout requires at least 2D region, got shape={region_shape}")
        if len(region_shape) > 2 and any(int(dim) != 1 for dim in region_shape[:-2]):
            raise ValueError(f"make_sqmma_swizzled_layout only supports BufferRegion with leading singleton dims, got shape={region_shape}")

        m_dim = int(region_shape[-2])
        n_dim = int(region_shape[-1])
        if continuity is None:
            continuity = n_dim
        base = _ffi_api.make_sqmma_swizzled_layout(
            m_dim,
            n_dim,
            continuity,
            int(tvm.DataType(buffer.buffer.dtype).bits),
            k_major,
        )

        target_shape = list(buffer.buffer.shape)
        if len(target_shape) == 2:
            return base

        prefix_rank = len(target_shape) - 2

        def forward_fn(*indices):
            mapped = list(base.map_forward_index([indices[-2], indices[-1]]))
            return list(indices[:prefix_rank]) + mapped

        return Layout(target_shape, forward_fn)

    # Fallback to generic behavior for other BufferLikeType wrappers.
    _, shape, _ = _get_buffer_info(buffer)
    stride, continuous = _get_stride_continuous(buffer)
    element_size = _get_element_size(buffer)
    if continuity is None:
        continuity = continuous
    base = _ffi_api.make_sqmma_swizzled_layout(
        stride,
        continuous,
        continuity,
        element_size,
        k_major,
    )
    return base.reshape(shape)


# no-swizzle layout with 1:1 index mapping and no dimension change.
def make_no_swizzled_layout(buffer: tvm.tir.Buffer):
    if isinstance(buffer, tvm.tir.Buffer):
        target_shape = list(buffer.shape)

        def forward_fn(*indices):
            return list(indices)

        return Layout(target_shape, forward_fn)

    if isinstance(buffer, tvm.tir.BufferRegion):
        region_shape = [r.extent for r in buffer.region]
        if len(region_shape) < 2:
            raise ValueError(f"make_no_swizzled_layout requires at least 2D region, got shape={region_shape}")
        if len(region_shape) > 2 and any(int(dim) != 1 for dim in region_shape[:-2]):
            raise ValueError(
                f"make_no_swizzled_layout only supports BufferRegion with leading singleton dimensions, got shape={region_shape}"
            )

        target_shape = list(buffer.buffer.shape)
        if len(target_shape) < 2:
            raise ValueError(f"make_no_swizzled_layout requires underlying buffer to be at least 2D, got shape={target_shape}")

        m_dim, n_dim = region_shape[-2], region_shape[-1]
        base_layout = Layout([m_dim, n_dim], lambda i, j: [i, j])
        if len(target_shape) == 2:
            return base_layout

        prefix_rank = len(target_shape) - 2

        def forward_fn(*indices):
            mapped = list(base_layout.map_forward_index([indices[-2], indices[-1]]))
            return list(indices[:prefix_rank]) + mapped

        return Layout(target_shape, forward_fn)

    raise ValueError(f"Unsupported buffer type for make_no_swizzled_layout: {type(buffer)}")


# for TCGEN05MMA Intrinsics
def make_tcgen05mma_swizzled_layout(buffer: BufferLikeType, continuity: int = None, k_major: bool = True):
    _, shape, _ = _get_buffer_info(buffer)
    stride, continuous = _get_stride_continuous(buffer)
    element_size = _get_element_size(buffer)
    if continuity is None:
        continuity = continuous
    base = _ffi_api.make_tcgen05mma_swizzled_layout(
        stride,
        continuous,
        continuity,
        element_size,
        k_major,
    )
    return base.reshape(shape)


# swizzle 128B
def make_full_bank_swizzled_layout(buffer: BufferLikeType):
    """
    Args:
        buffer: BufferLikeType
    Examples:
        make_full_bank_swizzled_layout(buffer)
    """
    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_full_bank_swizzled_layout(buf)


# swizzle 64B
def make_half_bank_swizzled_layout(buffer: BufferLikeType):
    """
    Args:
        buffer: BufferLikeType
    Examples:
        make_half_bank_swizzled_layout(buffer)
    """
    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_half_bank_swizzled_layout(buf)


# swizzle 32B
def make_quarter_bank_swizzled_layout(buffer: BufferLikeType):
    """
    Args:
        buffer: BufferLikeType
    Examples:
        make_quarter_bank_swizzled_layout(buffer)
    """
    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_quarter_bank_swizzled_layout(buf)


def make_linear_layout(buffer_or_load_or_region: BufferLikeType):
    """
    Create a row-major linear layout for any dimension.

    Args:
        buffer_or_load_or_region: BufferLikeType

    Returns:
        Layout: A row-major linear layout
    """
    _, shape, _ = _get_buffer_info(buffer_or_load_or_region)
    return _ffi_api.make_linear_layout(list(shape))


def make_gemm_layout_linear(stride: int, continuous: int):
    """Create a 2D row-major GEMM layout by (stride, continuous)."""
    return _ffi_api.make_gemm_layout_linear(int(stride), int(continuous))


def make_gemm_fragment_c_linear(block_m: int, block_n: int, block_size: int):
    """Create the PH1 FMA fragment layout for C."""
    return _ffi_api.make_gemm_fragment_c_linear(int(block_m), int(block_n), int(block_size))


def make_ph_sqmma_fragment_c(
    block_m: int,
    block_n: int,
    warp_m: int,
    warp_n: int,
    element_size: int,
    inst_shape: tuple[int, int, int] | list[int],
):
    """Create PH1 SQMMA fragment layout for C."""
    if len(inst_shape) != 3:
        raise ValueError(f"inst_shape must be [M, N, K], got {inst_shape}")
    return _ffi_api.make_ph_sqmma_fragment_c(
        int(block_m),
        int(block_n),
        int(warp_m),
        int(warp_n),
        int(element_size),
        [int(inst_shape[0]), int(inst_shape[1]), int(inst_shape[2])],
    )


def make_gemm_fragment_8x8():
    """
    Create a standard 8x8 GEMM fragment layout for ldmatrix/stmatrix.

    This layout matches the warp-level matrix multiplication pattern used in tensor cores.

    Returns:
        Fragment: An 8x8 fragment layout
    """
    return _ffi_api.make_gemm_fragment_8x8()


def make_gemm_fragment_8x8_transposed():
    """
    Create a transposed 8x8 GEMM fragment layout for ldmatrix/stmatrix.

    This layout is the transposed version of make_gemm_fragment_8x8, useful for
    different access patterns in matrix operations.

    Returns:
        Fragment: A transposed 8x8 fragment layout
    """
    return _ffi_api.make_gemm_fragment_8x8_transposed()


def make_fully_replicated_layout_fragment(buffer: BufferLikeType, threads: int):
    """
    Create a fully replicated layout for a fragment buffer.

    A fully replicated fragment means all threads hold identical copies of the
    entire buffer. This is useful for index buffers or masks that need to be
    accessed uniformly across all threads.

    Args:
        buffer: BufferLikeType to get shape information
        threads: Number of threads (replicate extent)

    Returns:
        Fragment: A fully replicated layout where each thread has a complete copy

    Example:
        >>> C_local = T.alloc_fragment((2,), T.float32)
        >>> layout = make_fully_replicated_layout_fragment(C_local, 256)
        >>> T.annotate_layout({C_local: layout})
    """
    _, shape, _ = _get_buffer_info(buffer)
    return _ffi_api.make_fully_replicated_layout_fragment(list(shape), threads)
