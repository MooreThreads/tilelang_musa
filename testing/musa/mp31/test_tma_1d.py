import re
import pytest
import torch
import tilelang
import tilelang.language as T

tilelang.disable_cache()


@tilelang.jit(target="musa")
def tma_copy_1d(A, block_n=128, dtype="float32"):
    N = T.const("N")
    A: T.Tensor[[N], dtype]
    C = T.empty((N,), dtype)

    with T.Kernel(T.ceildiv(N, block_n), threads=128) as bx:
        tile = T.alloc_shared((block_n,), dtype)
        T.copy(A[bx * block_n], tile, disable_tma=False)
        T.copy(tile, C[bx * block_n], disable_tma=True)

    return C


@pytest.mark.parametrize("N, BLOCK_N", [
    (8192, 128),
    (4096, 128),
    (16384, 256),
])
def test_tma_1d(N, BLOCK_N):
    kernel = tma_copy_1d.compile(N=N, block_n=BLOCK_N)
    code = kernel.get_kernel_source()
    tma_load_pattern = rf"tl::tma_load.*{BLOCK_N}"
    assert re.search(tma_load_pattern, code), (
        f"tl::tma_load with BLOCK_N={BLOCK_N} not found in generated code"
    )

    a = torch.randn(N, device="musa", dtype=torch.float32)
    c = kernel(a)
    torch.testing.assert_close(c, a, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_tma_1d(8192, 128)
    print("Test completed successfully!")
