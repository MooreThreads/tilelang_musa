import argparse
import tilelang
import tilelang.language as T
import torch
from tilelang.utils.tensor import map_torch_type
from tilelang.tileop.base import GemmWarpPolicy

TARGET = "musa"
DEVICE = "musa"
tilelang.disable_cache()


@tilelang.jit(target=TARGET)
def matmul_trans_b(A,
                   B,
                   block_M=128,
                   block_N=128,
                   block_K=64,
                   dtype="float16",
                   accum_dtype="float",
                   num_warp=4,
                   policy="square"):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[N, K], dtype]
    C = T.empty((M, N), dtype)

    thread_per_block = num_warp * 32
    if policy == "square":
        warp_policy = GemmWarpPolicy.Square
    elif policy == "m":
        warp_policy = GemmWarpPolicy.FullRow
    elif policy == "n":
        warp_policy = GemmWarpPolicy.FullCol
    else:
        raise ValueError(f"Unsupported policy: {policy}")

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
            T.gemm(A_shared, B_shared, C_local, transpose_B=True, policy=warp_policy)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def run(M, N, K, bm, bn, bk, dtype, acc_type, num_warp, policy, verbose):
    if verbose >= 1:
        print("Compiling matmul kernel...")

    kernel = matmul_trans_b.compile(
        M=M,
        N=N,
        K=K,
        block_M=bm,
        block_N=bn,
        block_K=bk,
        dtype=dtype,
        accum_dtype=acc_type,
        num_warp=num_warp,
        policy=policy,
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
    parser.add_argument("-policy", type=str, choices=["m", "n", "square"], default="square")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v: info logs, -vv: add kernel source",
    )
    args, _ = parser.parse_known_args()
    run(
        args.m,
        args.n,
        args.k,
        args.bm,
        args.bn,
        args.bk,
        args.dtype,
        args.acctype,
        args.warp,
        args.policy,
        args.verbose,
    )


if __name__ == "__main__":
    main()
