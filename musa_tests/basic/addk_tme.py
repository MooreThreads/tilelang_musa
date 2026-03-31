import tilelang
import tilelang.language as T

tilelang.disable_cache()


# 计算 MxN 大小的矩阵加一, BLOCK 大小为 128x128
M = 4096
N = 4096

# T.copy 降低为 TMA
def tma_add_one(M: int, N: int,
                BLOCK_M: int = 128, BLOCK_N: int = 128,
                dtype: str = "float32"):
    @T.prim_func
    def kernel(A: T.Tensor((M, N), dtype),
               C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=128) as (bx, by):
            tile = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)

            # --- TMA load: GMEM -> SMEM ---
            # disable_tma=False(默认)表示允许使用 TMA
            T.copy(A[by * BLOCK_M, bx * BLOCK_N], tile, disable_tma=False)

            # 在共享内存做非常简单的逐元素计算(不涉及 MMA/WGMMA)
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                tile[i, j] = tile[i, j] + 1

            # --- TMA store: SMEM -> GMEM ---
            T.copy(tile, C[by * BLOCK_M, bx * BLOCK_N], disable_tma=False)

    return kernel

program = tma_add_one(M, N)
kernel = tilelang.compile(program, out_idx=-1, target="musa", execution_backend="cython", verbose=True)
print(kernel.get_kernel_source())


import torch
A = torch.randn(M, N, device="musa", dtype=torch.float32)
C = kernel(A)
print(C)
torch.testing.assert_close(C, A + 1, rtol=1e-6, atol=1e-6)
print("OK")
