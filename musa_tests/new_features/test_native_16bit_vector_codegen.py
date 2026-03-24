import pytest
import torch
import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang import tvm

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def require_musa():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA is not available")


def scalar_constant(dtype: str, value: int):
    if dtype == "float16":
        return T.float16(value)
    if dtype == "bfloat16":
        return T.bfloat16(value)
    raise ValueError(f"Unsupported dtype: {dtype}")


def vector_let_load_store(dtype: str, lanes: int):

    @T.prim_func
    def main(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (1, lanes * 2), dtype=dtype, align=16)

        for _blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for _threadIdx in T.thread_binding(1, thread="threadIdx.x"):
                b = A[0, 0:lanes]
                A[0, lanes:lanes * 2] = b

    return main


def vector_broadcast_store(dtype: str, lanes: int):

    @T.prim_func
    def main(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (1, lanes), dtype=dtype, align=16)

        for _blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for _threadIdx in T.thread_binding(1, thread="threadIdx.x"):
                A[0, 0:lanes] = T.Broadcast(scalar_constant(dtype, 1), lanes)

    return main


def vector_ramp_store(dtype: str, lanes: int):

    @T.prim_func
    def main(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (1, lanes), dtype=dtype, align=16)

        for _blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for _threadIdx in T.thread_binding(1, thread="threadIdx.x"):
                A[0, 0:lanes] = tvm.tir.Ramp(
                    scalar_constant(dtype, 1),
                    scalar_constant(dtype, 1),
                    lanes,
                )

    return main


def vector_non_ramp_index_load(dtype: str, lanes: int):

    @T.prim_func
    def main(src_ptr: T.handle, out_ptr: T.handle):
        src = T.match_buffer(src_ptr, (1, 1), dtype=dtype, align=16)
        out = T.match_buffer(out_ptr, (1, lanes), dtype=dtype, align=16)

        for _blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for _threadIdx in T.thread_binding(1, thread="threadIdx.x"):
                idx = tvm.tir.Broadcast(T.int32(0), lanes)
                out[0, 0:lanes] = src[0, idx]

    return main


@tilelang.jit(target="musa", out_idx=[1], pass_configs=PASS_CONFIGS)
def vector_binary_op(dtype: str, lanes: int):

    @T.prim_func
    def main(
            src: T.Tensor((1, lanes * 2), dtype),
            out: T.Tensor((1, lanes * 2), dtype),
    ):
        with T.Kernel(1, threads=1) as _:
            lhs = src[0, 0:lanes]
            rhs = src[0, lanes:lanes * 2]
            out[0, 0:lanes] = lhs + rhs
            out[0, lanes:lanes * 2] = lhs - rhs

    return main


def reference_vector_binary_op(src: torch.Tensor, lanes: int) -> torch.Tensor:
    lhs = src[:, :lanes].cpu().float()
    rhs = src[:, lanes:lanes * 2].cpu().float()
    return torch.cat([lhs + rhs, lhs - rhs], dim=1).to(device=src.device, dtype=src.dtype)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "dtype, lanes, expected_type, old_packed_type",
    [
        ("float16", 2, "tl_h2", "uint1"),
        ("float16", 4, "tl_h4", "uint2"),
        ("float16", 8, "tl_h8", "uint4"),
        ("bfloat16", 2, "tl_bf2", "uint1"),
        ("bfloat16", 4, "tl_bf4", "uint2"),
        ("bfloat16", 8, "tl_bf8", "uint4"),
    ],
)
def test_native_16bit_vector_alias_is_emitted(dtype, lanes, expected_type, old_packed_type):
    program = vector_let_load_store(dtype, lanes)
    kernel = tilelang.compile(program, target="musa")
    source = kernel.get_kernel_source()

    assert f"{expected_type} b" in source
    assert f"{old_packed_type} b" not in source


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "dtype, lanes, expected_type",
    [
        ("float16", 2, "tl_h2"),
        ("float16", 4, "tl_h4"),
        ("float16", 8, "tl_h8"),
        ("bfloat16", 2, "tl_bf2"),
        ("bfloat16", 4, "tl_bf4"),
        ("bfloat16", 8, "tl_bf8"),
    ],
)
def test_native_16bit_vector_broadcast_is_emitted(dtype, lanes, expected_type):
    program = vector_broadcast_store(dtype, lanes)
    kernel = tilelang.compile(program, target="musa")
    source = kernel.get_kernel_source()

    assert f"({expected_type}{{" in source
    assert "make_" not in source


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "dtype, lanes, expected_type",
    [
        ("float16", 2, "tl_h2"),
        ("float16", 4, "tl_h4"),
        ("bfloat16", 2, "tl_bf2"),
        ("bfloat16", 4, "tl_bf4"),
    ],
)
def test_native_16bit_vector_ramp_is_emitted(dtype, lanes, expected_type):
    program = vector_ramp_store(dtype, lanes)
    kernel = tilelang.compile(program, target="musa")
    source = kernel.get_kernel_source()

    assert f"({expected_type}{{" in source
    assert "make_" not in source


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "dtype, lanes, expected_type",
    [
        ("float16", 2, "tl_h2"),
        ("float16", 4, "tl_h4"),
        ("float16", 8, "tl_h8"),
        ("bfloat16", 2, "tl_bf2"),
        ("bfloat16", 4, "tl_bf4"),
        ("bfloat16", 8, "tl_bf8"),
    ],
)
def test_native_16bit_vector_non_ramp_load_rebuilds_vector(dtype, lanes, expected_type):
    program = vector_non_ramp_index_load(dtype, lanes)
    kernel = tilelang.compile(program, target="musa")
    source = kernel.get_kernel_source()
    normalized_source = source.replace(" ", "")

    assert f"({expected_type}{{" in source
    assert f"*({expected_type}*)(src+" not in normalized_source


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "dtype, lanes, old_lane_access_pattern",
    [
        ("float16", 2, "((half2*)(&("),
        ("float16", 4, "((half2*)(&("),
        ("float16", 8, "((half2*)(&("),
        ("bfloat16", 2, "((mt_bfloat162*)(&("),
        ("bfloat16", 4, "((mt_bfloat162*)(&("),
        ("bfloat16", 8, "((mt_bfloat162*)(&("),
    ],
)
def test_native_16bit_vector_lane_access_uses_native_indexing(dtype, lanes,
                                                              old_lane_access_pattern):
    kernel = vector_binary_op(dtype, lanes)
    source = kernel.get_kernel_source()

    assert "lhs[0]" in source
    assert "__1[0]" in source
    assert old_lane_access_pattern not in source


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "dtype, torch_dtype, lanes",
    [
        ("float16", torch.float16, 2),
        ("float16", torch.float16, 4),
        ("float16", torch.float16, 8),
        ("bfloat16", torch.bfloat16, 2),
        ("bfloat16", torch.bfloat16, 4),
        ("bfloat16", torch.bfloat16, 8),
    ],
)
def test_native_16bit_vector_numerical(dtype, torch_dtype, lanes):
    require_musa()

    src = torch.arange(1, lanes * 2 + 1, device="musa", dtype=torch.float32).reshape(1, lanes * 2)
    src = src.to(torch_dtype)

    kernel = vector_binary_op(dtype, lanes)
    out = kernel(src)
    if isinstance(out, (tuple, list)):
        out = out[0]

    ref = reference_vector_binary_op(src, lanes)
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)
