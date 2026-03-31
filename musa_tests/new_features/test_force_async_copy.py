import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

PASS_CONFIGS_DISABLE_THREAD_STORAGE_SYNC = {
    **PASS_CONFIGS,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}


def require_musa():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA is not available")


@tilelang.jit(target="musa", out_idx=[1], pass_configs=PASS_CONFIGS)
def kernel_with_vectorized_scalar_force_async_copy_to_shared():

    @T.prim_func
    def main(
        src: T.Tensor([4], T.float32),
        out: T.Tensor([4], T.float32),
    ):
        with T.Kernel(1, threads=1) as _:
            src_shared = T.alloc_shared([4], T.float32)
            for v in T.vectorized(4):
                T.copy(src[v], src_shared[v], force_async_copy=True)
            T.copy(src_shared, out)

    return main


@tilelang.jit(target="musa", out_idx=[1], pass_configs=PASS_CONFIGS_DISABLE_THREAD_STORAGE_SYNC)
def kernel_with_vectorized_scalar_force_async_copy_to_shared_disable_thread_storage_sync():

    @T.prim_func
    def main(
        src: T.Tensor([4], T.float32),
        out: T.Tensor([4], T.float32),
    ):
        with T.Kernel(1, threads=1) as _:
            src_shared = T.alloc_shared([4], T.float32)
            for v in T.vectorized(4):
                T.copy(src[v], src_shared[v], force_async_copy=True)
            T.copy(src_shared, out)

    return main


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_vectorized_scalar_force_async_copy_to_shared_numerical():
    require_musa()

    src = torch.tensor([1.0, -2.0, 3.0, -4.0], device="musa", dtype=torch.float32)
    kernel = kernel_with_vectorized_scalar_force_async_copy_to_shared()
    out = kernel(src)
    if isinstance(out, (tuple, list)):
        out = out[0]

    torch.testing.assert_close(out, src, rtol=0.0, atol=0.0)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_vectorized_scalar_force_async_copy_to_shared_source():
    code = kernel_with_vectorized_scalar_force_async_copy_to_shared().get_kernel_source()

    assert "tl::cp_async_gs<16>" in code
    assert "tl::cp_async_wait<0>();" in code


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_vectorized_scalar_force_async_copy_to_shared_source_disable_thread_storage_sync():
    code = kernel_with_vectorized_scalar_force_async_copy_to_shared_disable_thread_storage_sync().get_kernel_source()

    assert "tl::cp_async_gs<16>" in code
    assert "tl::cp_async_wait<0>();" not in code
