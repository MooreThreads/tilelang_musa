import torch
import tilelang
from tilelang import language as T

# Keep the generated kernel deterministic across runs.
tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(target="musa", pass_configs=PASS_CONFIGS, verbose=True)
def parallel_shared_gemm(
    M, N, K,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 64,
    num_stages: int = 2,
    threads: int = 256,
):
    """GEMM that loads tiles into shared memory via Parallel instead of copy."""
    dtype = "bfloat16"
    accum_dtype = "float"

    num_stages = 0

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (
            bx,
            by,
        ):
            # Shared-memory tiles. B_shared keeps B in column-major so we only need a transpose flag in gemm.
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.fill(C_local, 0)
            k_tiles = T.ceildiv(K, block_K)

            for k_tile in T.Pipelined(k_tiles, num_stages=num_stages):
                # Load A tile from GMEM to SMEM with Parallel guards.
                for i, kk in T.Parallel(block_M, block_K):
                    global_m = by * block_M + i
                    global_k = k_tile * block_K + kk
                    A_shared[i, kk] = T.if_then_else(
                        (global_m < M) & (global_k < K),
                        A[global_m, global_k],
                        0,
                    )

                # Load B tile from GMEM to SMEM with Parallel guards.
                for j, kk in T.Parallel(block_N, block_K):
                    global_n = bx * block_N + j
                    global_k = k_tile * block_K + kk
                    B_shared[j, kk] = T.if_then_else(
                        (global_n < N) & (global_k < K),
                        B[global_n, global_k],
                        0,
                    )

                # Compute partial C.
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


M, N, K = 512, 512, 512


kernel = parallel_shared_gemm(M, N, K)
kernel.show_source()



device = "musa"
a = torch.randn((M, K), device=device, dtype=torch.bfloat16)
b = torch.randn((N, K), device=device, dtype=torch.bfloat16)
c = torch.empty((M, N), device=device, dtype=torch.bfloat16)
kernel(a, b, c)

c_ref = a @ b.T

torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)
print("pass")
