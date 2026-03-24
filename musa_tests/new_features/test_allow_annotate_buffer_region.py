import pytest
import tilelang
import tilelang.testing
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
def two_gemm_layout_kernel(
    M,
    N,
    K,
    *,
    threads=128,
):
    dtype = "bfloat16"
    accum_dtype = "float"

    @T.prim_func
    def main(
            Q: T.Tensor((M, K), dtype),
            K_in: T.Tensor((N, K), dtype),
            V: T.Tensor((N, K), dtype),
            O: T.Tensor((M, K), dtype),
    ):
        with T.Kernel(1, threads=threads) as _:
            Q_shared = T.alloc_shared((M, K), dtype)
            KV_shared = T.alloc_shared((2, N, K), dtype)
            S_shared = T.alloc_shared((M, N), dtype)
            acc_s = T.alloc_fragment((M, N), accum_dtype)
            acc_o = T.alloc_fragment((M, K), accum_dtype)
            T.fill(acc_s, 0)
            T.fill(acc_o, 0)

            T.copy(Q, Q_shared)
            T.annotate_layout(
                {
                    KV_shared[0, :, :]:
                        tilelang.layout.make_sqmma_swizzled_layout(
                            KV_shared[0, :, :], k_major=True)
                },
                allow_reannotation=True,
                allow_buffer_region=True,
            )
            T.copy(K_in, KV_shared[0, :, :])
            T.gemm(Q_shared, KV_shared[0, :, :], acc_s, transpose_B=True)

            T.copy(acc_s, S_shared)
            T.annotate_layout(
                {
                    KV_shared[1, :, :]:
                        tilelang.layout.make_sqmma_swizzled_layout(
                            KV_shared[1, :, :], k_major=False)
                },
                allow_reannotation=True,
                allow_buffer_region=True,
            )
            T.copy(V, KV_shared[1, :, :])
            T.gemm(S_shared, KV_shared[1, :, :], acc_o)

            T.copy(acc_o, O)

    return main


def ref_two_gemm(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    s = (Q.float() @ K.float().transpose(0, 1)).to(torch.bfloat16)
    o = s.float() @ V.float()
    return o.to(torch.bfloat16)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "M, N, K, dtype, threads",
    [
        (64, 64, 64, torch.bfloat16, 128),
        (64, 64, 256, torch.bfloat16, 128),
    ],
)
def test_two_gemm_layout(M, N, K, dtype, threads):
    torch.random.manual_seed(0)
    device = get_test_device()
    if not device:
        pytest.skip("Neither MUSA nor CUDA is available")

    q = torch.randn((M, K), dtype=dtype, device=device)
    k = torch.randn((N, K), dtype=dtype, device=device)
    v = torch.randn((N, K), dtype=dtype, device=device)

    kernel = two_gemm_layout_kernel(M, N, K, threads=threads)
    out = kernel(q, k, v)

    ref_out = ref_two_gemm(q.cpu(), k.cpu(), v.cpu()).to(device)
    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
