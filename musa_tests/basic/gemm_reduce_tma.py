# musa_tests/basic/gemm_reduce.py
import torch
import tilelang
from tilelang import language as T

tilelang.disable_cache()


@tilelang.jit(target="musa", out_idx=[-1])
def gemm_reduce(M, N, K, threads=256):

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
            T.reduce_sum(acc, sum, dim=1, clear=True)

            T.copy(sum, Out)

    return main


def gemm_reduce_ref(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    A: [ROWS, DEPTH], B: [COLS, DEPTH]
    returns: Out [ROWS], where Out[r] = sum_k (A[r] · B[k])
    """
    # 确保 dtype/设备与你的 TileLang kernel 一致
    scores = torch.matmul(A, B.transpose(0, 1))  # [ROWS, COLS]
    scores = scores.sum(dim=1)  # [ROWS]
    return scores


if __name__ == "__main__":
    # MP31 SQMMA requires tile-friendly sizes: M multiple of 16, N multiple of 8
    M, N, K = 32, 64, 64
    kernel = gemm_reduce(M, N, K)
    print(kernel.get_kernel_source())

    A = torch.randn(M, K, device="musa", dtype=torch.float16)
    B = torch.randn(N, K, device="musa", dtype=torch.float16)
    out_ref = gemm_reduce_ref(A, B).float()  # 可转 float 便于比较
    out = kernel(A, B)

    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    print("pass")
