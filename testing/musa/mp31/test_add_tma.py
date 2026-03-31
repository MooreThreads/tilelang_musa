import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()


@tilelang.jit(target="musa", verbose=True)
def tma_add_one(A, block_m: int = 128, block_n: int = 128, dtype: str = "float32"):
    M, N = T.const("M N")
    A: T.Tensor[[M, N], dtype]
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=128) as (bx, by):
        tile = T.alloc_shared((block_m, block_n), dtype)

        # TMA load: GMEM -> SMEM.
        T.copy(A[by * block_m, bx * block_n], tile, disable_tma=False)

        # Simple elementwise compute on shared memory.
        for i, j in T.Parallel(block_m, block_n):
            tile[i, j] = tile[i, j] + 1

        # TMA store: SMEM -> GMEM.
        T.copy(tile, C[by * block_m, bx * block_n], disable_tma=False)

    return C


def ref_program(x):
    return x + 1


def test_tma_add_one():
    M = 4096
    N = 4096
    kernel = tma_add_one.compile(M=M, N=N)
    a = torch.randn(M, N, device="musa", dtype=torch.float32)
    c = kernel(a)
    torch.testing.assert_close(c, ref_program(a), rtol=1e-6, atol=1e-6)


def main():
    M = 4096
    N = 4096
    kernel = tma_add_one.compile(M=M, N=N)
    print(kernel.get_kernel_source())

    a = torch.randn(M, N, device="musa", dtype=torch.float32)
    c = kernel(a)
    torch.testing.assert_close(c, ref_program(a), rtol=1e-6, atol=1e-6)
    print("pass!")


if __name__ == "__main__":
    main()
