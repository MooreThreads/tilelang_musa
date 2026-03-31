import pytest
import re
import torch
import tilelang
import tilelang.language as T


@pytest.mark.parametrize("N, BLOCK_N", [
    (8192, 128),
    (4096, 128),
    (16384, 256),
])
def test_tma_1d(N, BLOCK_N):
    """
    Test 1D TMA load operation.
    Copies data from global memory to shared memory using TMA,
    then stores back to verify correctness.
    """

    dtype = "float32"

    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=128) as (bx,):
            tile = T.alloc_shared((BLOCK_N,), dtype)

            # --- TMA load: GMEM -> SMEM (1D) ---
            # disable_tma=False (default) enables TMA
            T.copy(A[bx * BLOCK_N], tile, disable_tma=False)

            # --- Regular store: SMEM -> GMEM (1D) ---
            T.copy(tile, C[bx * BLOCK_N], disable_tma=True)

    # Compile kernel
    compiled_kernel = tilelang.compile(
        kernel,
        out_idx=-1,
        target="musa",
        execution_backend="cython",
        verbose=False,
    )

    # Verify TMA load in generated code
    code = compiled_kernel.get_kernel_source()
    tma_load_pattern = r'tl::tma_load.*' + str(BLOCK_N)

    assert re.search(tma_load_pattern, code), \
        f"tl::tma_load with BLOCK_N={BLOCK_N} not found in generated code"

    # Test execution correctness
    A = torch.randn(N, device="musa", dtype=torch.float32)
    C = compiled_kernel(A)

    torch.testing.assert_close(C, A, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    # Run single test for debugging
    tilelang.disable_cache()
    test_tma_1d(8192, 128)
    print("Test completed successfully!")
