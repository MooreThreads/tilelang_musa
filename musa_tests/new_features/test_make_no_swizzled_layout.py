import pytest
import tilelang
import torch
from tilelang import language as T

tilelang.disable_cache()


def get_test_device() -> str:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    return ""


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def no_swizzle_buffer_kernel(
    M,
    N,
    *,
    threads=128,
):
    dtype = "bfloat16"

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            O: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, threads=threads) as _:
            A_shared = T.alloc_shared((M, N), dtype)
            T.annotate_layout({A_shared: tilelang.layout.make_no_swizzled_layout(A_shared)})
            T.copy(A, A_shared)
            for i, j in T.Parallel(M, N):
                O[i, j] = A_shared[i, j]

    return main


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def no_swizzle_buffer_region_kernel(
    M,
    N,
    *,
    threads=128,
):
    dtype = "bfloat16"

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            O: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, threads=threads) as _:
            AB_shared = T.alloc_shared((2, M, N), dtype)
            T.annotate_layout(
                {AB_shared[0, :, :]: tilelang.layout.make_no_swizzled_layout(AB_shared[0, :, :])},
                allow_reannotation=True,
                allow_buffer_region=True,
            )
            T.copy(A, AB_shared[0, :, :])
            T.annotate_layout(
                {AB_shared[1, :, :]: tilelang.layout.make_no_swizzled_layout(AB_shared[1, :, :])},
                allow_reannotation=True,
                allow_buffer_region=True,
            )
            T.copy(B, AB_shared[1, :, :])
            for i, j in T.Parallel(M, N):
                O[i, j] = AB_shared[0, i, j] + AB_shared[1, i, j]

    return main


@pytest.mark.parametrize(
    "M, N, dtype, threads",
    [
        (64, 64, torch.bfloat16, 128),
    ],
)
def test_no_swizzle_layout_buffer(M, N, dtype, threads):
    torch.random.manual_seed(0)
    device = get_test_device()
    if not device:
        pytest.skip("Neither MUSA nor CUDA is available")

    a = torch.randn((M, N), dtype=dtype, device=device)
    kernel = no_swizzle_buffer_kernel(M, N, threads=threads)
    out = kernel(a)

    torch.testing.assert_close(out, a, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "M, N, dtype, threads",
    [
        (64, 64, torch.bfloat16, 128),
        (64, 256, torch.bfloat16, 128),
    ],
)
def test_no_swizzle_layout_buffer_region(M, N, dtype, threads):
    torch.random.manual_seed(0)
    device = get_test_device()
    if not device:
        pytest.skip("Neither MUSA nor CUDA is available")

    a = torch.randn((M, N), dtype=dtype, device=device)
    b = torch.randn((M, N), dtype=dtype, device=device)
    kernel = no_swizzle_buffer_region_kernel(M, N, threads=threads)
    out = kernel(a, b)

    torch.testing.assert_close(out, a + b, rtol=1e-2, atol=1e-2)
