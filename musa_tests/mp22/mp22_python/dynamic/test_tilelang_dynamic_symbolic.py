import pytest
import torch
import torch.backends
from tilelang import tvm as tvm
import tilelang.testing
from tvm import DataType
import tilelang.language as T
from tilelang.intrinsics.utils import get_swizzle_layout
from tilelang.intrinsics.mma_macro_generator import (TensorCoreIntrinEmitter)

tilelang.testing.set_random_seed(0)


def make_swizzle_layout(shared_buf):
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    can_swizzle = shape[-1] * DataType(dtype).bits == 512
    if not can_swizzle:
        return T.Layout(shape, lambda *args: args)

    def transform_func(i, j):
        new_warp_i, new_warp_j = get_swizzle_layout(i, j, shape[-1], dtype)
        return [new_warp_i, new_warp_j]

    return T.Layout(shape, transform_func)


def tl_matmul_block(
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    M = tvm.te.var("m")
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(A: T.Tensor(A_shape, in_dtype), B: T.Tensor(B_shape, in_dtype), C: T.Tensor(
        (M, N), out_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def assert_tl_matmul_block_correctness(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
):
    program = tl_matmul_block(
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(program, out_idx=[2])

    A = torch.rand(M, K, device="musa", dtype=getattr(torch, in_dtype))
    B = torch.rand(N, K, device="musa", dtype=getattr(torch, in_dtype))
    C = torch.zeros(M, N, device="musa", dtype=getattr(torch, out_dtype))

    C = kernel(A, B)

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    # Get Reference Result
    ref_c = ref_program(A, B)

    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def tl_matmul_block_all_dynamic(
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    M = tvm.te.var("m")
    N = tvm.te.var("n")
    K = tvm.te.var("k")

    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(A: T.Tensor(A_shape, in_dtype), B: T.Tensor(B_shape, in_dtype), C: T.Tensor(
        (M, N), out_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def assert_tl_matmul_block_all_dynamic_correctness(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
):
    program = tl_matmul_block_all_dynamic(
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(program)

    if trans_A:
        A = torch.rand(K, M, device="musa", dtype=getattr(torch, in_dtype))
    else:
        A = torch.rand(M, K, device="musa", dtype=getattr(torch, in_dtype))
    if trans_B:
        B = torch.rand(N, K, device="musa", dtype=getattr(torch, in_dtype))
    else:
        B = torch.rand(K, N, device="musa", dtype=getattr(torch, in_dtype))
    C = torch.zeros(M, N, device="musa", dtype=getattr(torch, out_dtype))

    kernel(A, B, C)

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    # Get Reference Result
    ref_c = ref_program(A, B)

    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def assert_tl_matmul_block_all_dynamic_correctness_with_pass_config(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
    dynamic_alignment=8,
):
    program = tl_matmul_block_all_dynamic(
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_DYNAMIC_TAIL_SPLIT: dynamic_alignment != 0,
        tilelang.PassConfigKey.TL_DYNAMIC_ALIGNMENT: dynamic_alignment
    }
    if M % 64 == 0 or N % 64 == 0 or K % 64 != 0:
        # workaround for hopper tma lower pass
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER] = True
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED] = True

    kernel = tilelang.compile(program, pass_configs=pass_configs)

    if trans_A:
        A = torch.rand(K, M, device="musa", dtype=getattr(torch, in_dtype))
    else:
        A = torch.rand(M, K, device="musa", dtype=getattr(torch, in_dtype))
    if trans_B:
        B = torch.rand(N, K, device="musa", dtype=getattr(torch, in_dtype))
    else:
        B = torch.rand(K, N, device="musa", dtype=getattr(torch, in_dtype))
    C = torch.zeros(M, N, device="musa", dtype=getattr(torch, out_dtype))

    kernel(A, B, C)

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    # Get Reference Result
    ref_c = ref_program(A, B)

    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_assert_tl_matmul_block():
    assert_tl_matmul_block_correctness(128, 128, 128, False, False, "float16", "float16", "float32",
                                       64, 64, 32)
    assert_tl_matmul_block_correctness(67, 128, 128, False, False, "float16", "float16", "float32",
                                       64, 64, 32)
    assert_tl_matmul_block_correctness(36, 128, 128, False, False, "float16", "float16", "float32",
                                       64, 64, 32)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_assert_tl_matmul_block_all_dynamic():
    assert_tl_matmul_block_all_dynamic_correctness(128, 128, 128, False, False, "float16",
                                                   "float16", "float32", 64, 64, 32)
    assert_tl_matmul_block_all_dynamic_correctness(67, 128, 128, False, False, "float16", "float16",
                                                   "float32", 64, 64, 32)
    assert_tl_matmul_block_all_dynamic_correctness(36, 128, 128, False, False, "float16", "float16",
                                                   "float32", 64, 64, 32)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_assert_tl_matmul_block_all_dynamic_with_pass_config():
    assert_tl_matmul_block_all_dynamic_correctness_with_pass_config(
        128,
        128,
        128,
        False,
        False,
        "float16",
        "float16",
        "float32",
        64,
        64,
        32,
        dynamic_alignment=8)
    assert_tl_matmul_block_all_dynamic_correctness_with_pass_config(
        64,
        128,
        128,
        False,
        False,
        "float16",
        "float16",
        "float32",
        64,
        64,
        32,
        dynamic_alignment=8)
    assert_tl_matmul_block_all_dynamic_correctness_with_pass_config(
        64, 128, 60, False, False, "float16", "float16", "float32", 64, 64, 32, dynamic_alignment=4)
    # Tail split is enabled with dynamic alignment 0
    assert_tl_matmul_block_all_dynamic_correctness_with_pass_config(
        64, 128, 64, False, False, "float16", "float16", "float32", 64, 64, 32, dynamic_alignment=0)


if __name__ == "__main__":
    tilelang.testing.main()
