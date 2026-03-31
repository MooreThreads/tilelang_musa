import argparse
import tilelang
import tilelang.language as T
import torch
from tilelang.utils.tensor import map_torch_type
from tilelang import tvm as tvm
from tvm.ir.instrument import PrintAfterAll
from tilelang.primitives.gemm.base import GemmWarpPolicy

TARGET = "musa"
DEVICE = "musa"


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype="float16",
    accum_dtype="float",
    num_warp=4,
    policy="square",
):

    thread_per_block = num_warp * 32
    if policy == "square":
        policy = GemmWarpPolicy.Square
    elif policy == "m":
        policy = GemmWarpPolicy.FullRow
    elif policy == "n":
        policy = GemmWarpPolicy.FullCol

    @T.prim_func
    def matmul_kernel(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_per_block) as (
                    bx,
                    by,
                ):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True, policy=policy)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return matmul_kernel


def run(M, N, K, bm, bn, bk, dtype, acc_type, num_warp, policy, verbose):

    program = matmul(
        M, N, K, bm, bn, bk, dtype=dtype, accum_dtype=acc_type, num_warp=num_warp, policy=policy)

    if verbose >= 1:
        print("Compiling matmul kernel...")
    instruments = [PrintAfterAll()] if verbose >= 2 else []

    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target=TARGET,
        execution_backend="cython",
        verbose=verbose >= 1,
        instruments=instruments,
    )

    if verbose >= 2:
        print(kernel.get_kernel_source())

    pt_type = map_torch_type(dtype)
    if pt_type is torch.float8_e4m3fn:
        a = torch.randint(
            low=-128, high=128, size=(M, K), device=DEVICE, dtype=torch.int8).to(pt_type)
        b = torch.randint(
            low=-128, high=128, size=(N, K), device=DEVICE, dtype=torch.int8).to(pt_type)
    else:
        a = torch.randn(M, K, device=DEVICE, dtype=pt_type)
        b = torch.randn(N, K, device=DEVICE, dtype=pt_type)
    if verbose >= 1:
        print("start kernel")
    c = kernel(a, b)
    ref_c = a @ b.T
    if pt_type is torch.float8_e4m3fn:
        torch.testing.assert_close(c.float(), ref_c.float(), rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    if verbose >= 1:
        print("tilelang kernel matches torch reference.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, default=512)
    parser.add_argument("-n", type=int, default=512)
    parser.add_argument("-k", type=int, default=512)
    parser.add_argument("-bm", type=int, default=128)
    parser.add_argument("-bn", type=int, default=128)
    parser.add_argument("-bk", type=int, default=64)
    parser.add_argument("-dtype", type=str, default="float16")
    parser.add_argument("-acctype", type=str, default="float32")
    parser.add_argument("-warp", type=int, default=4)
    parser.add_argument("-policy", type=str, choices=['m', 'n', 'square'], default="square")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v: info logs, -vv: add kernel source/instrumentation",
    )
    args, _ = parser.parse_known_args()
    M, N, K = args.m, args.n, args.k
    bm, bn, bk = args.bm, args.bn, args.bk
    dtype, acctype = args.dtype, args.acctype
    warp, policy = args.warp, args.policy
    verbose_level = args.verbose
    run(M, N, K, bm, bn, bk, dtype, acctype, warp, policy, verbose_level)


if __name__ == "__main__":
    tilelang.disable_cache()
    main()
