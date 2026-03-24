import argparse
import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()

TARGET = "musa"
DEVICE = "musa"


def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def matmul_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return matmul_kernel


def run(M, N, K, bm, bn, bk, dtype, acc_type, verbose):

    program = matmul(M, N, K, bm, bn, bk, dtype=dtype, accum_dtype=acc_type)

    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target=TARGET,
        execution_backend="cython",
        verbose=verbose,
    )

    if verbose:
        print(kernel.get_kernel_source())

    a = torch.randn(M, K, device=DEVICE, dtype=getattr(torch, dtype))
    b = torch.randn(K, N, device=DEVICE, dtype=getattr(torch, dtype))
    if verbose:
        print("start kernel")
    c = kernel(a, b)
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    if verbose:
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
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args, _ = parser.parse_known_args()
    M, N, K = args.m, args.n, args.k
    bm, bn, bk = args.bm, args.bn, args.bk
    dtype, acctype = args.dtype, args.acctype
    verbose = args.verbose
    run(M, N, K, bm, bn, bk, dtype, acctype, verbose)


if __name__ == "__main__":
    main()
