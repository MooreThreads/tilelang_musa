# musa_tests/basic/gemm_reduce.py
import torch
import tilelang
from tilelang import language as T

tilelang.disable_cache()


def gemm_reduce_max(M, N, K, threads=256):

    @T.prim_func
    def main(
            A: T.Tensor((M, K), "float16"),
            B: T.Tensor((N, K), "float16"),
            Out: T.Tensor((M,), "float"),
    ):
        with T.Kernel(1, threads=threads) as _:
            A_shared = T.alloc_shared((M, K), "float16")
            B_shared = T.alloc_shared((N, K), "float16")
            acc = T.alloc_fragment((M, N), "float")
            sum = T.alloc_fragment((M,), "float")

            T.clear(acc)
            T.copy(A, A_shared)
            T.copy(B, B_shared)

            T.gemm(A_shared, B_shared, acc, transpose_B=True)
            T.reduce_max(acc, sum, dim=1, clear=True)
            T.copy(sum, Out)

    return main


def gemm_reduce_max_ref(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(A, B.transpose(0, 1))
    scores = scores.max(dim=1).values
    return scores


if __name__ == "__main__":
    torch.random.manual_seed(2026)

    M, N, K = 128, 128, 64
    program = gemm_reduce_max(M, N, K)

    kernel = tilelang.compile(program, out_idx=[-1], verbose=True)
    print(kernel.get_kernel_source())

    A = torch.randn(M, K, device="musa", dtype=torch.float16)
    B = torch.randn(N, K, device="musa", dtype=torch.float16)
    out = kernel(A, B)
    out_ref = gemm_reduce_max_ref(A, B).float()

    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    print("pass")
