import pytest
import torch
import tilelang
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
def kernel_with_robust_load():

    @T.prim_func
    def main(
            src: T.Tensor([4], T.float32),
            out: T.Tensor([4], T.float32),
    ):
        with T.Kernel(1, threads=4) as _:
            tid = T.get_thread_binding()
            src_local = T.alloc_local([1], T.float32)
            robust_desc = T.make_robust_desc(T.address_of(src[1]), 8)
            T.copy(src[tid], src_local, src_robust_desc=robust_desc)
            out[tid] = src_local[0]

    return main


@tilelang.jit(target="musa", out_idx=[1], pass_configs=PASS_CONFIGS)
def kernel_with_robust_async_copy():

    @T.prim_func
    def main(
            src: T.Tensor([4], T.float32),
            out: T.Tensor([4], T.float32),
    ):
        with T.Kernel(1, threads=4) as _:
            src_shared = T.alloc_shared([4], T.float32)
            robust_desc = T.make_robust_desc(T.address_of(src[1]), 8)
            T.copy(src, src_shared, force_async_copy=True, src_robust_desc=robust_desc)
            T.copy(src_shared, out)

    return main


@tilelang.jit(target="musa", out_idx=[1], pass_configs=PASS_CONFIGS)
def kernel_with_zero_sized_robust_async_copy():

    @T.prim_func
    def main(
            src: T.Tensor([4], T.float32),
            out: T.Tensor([4], T.float32),
    ):
        with T.Kernel(1, threads=4) as _:
            src_shared = T.alloc_shared([4], T.float32)
            robust_desc = T.make_robust_desc(T.address_of(src[0]), 0)
            T.copy(src, src_shared, force_async_copy=True, src_robust_desc=robust_desc)
            T.copy(src_shared, out)

    return main


@tilelang.jit(target="musa", out_idx=[1], pass_configs=PASS_CONFIGS)
def kernel_with_scalar_robust_copy_to_shared():

    @T.prim_func
    def main(
            src: T.Tensor([4], T.float32),
            out: T.Tensor([4], T.float32),
    ):
        with T.Kernel(1, threads=4) as _:
            tid = T.get_thread_binding()
            src_shared = T.alloc_shared([4], T.float32)
            robust_desc = T.make_robust_desc(T.address_of(src[1]), 8)
            T.copy(src[tid], src_shared[tid], src_robust_desc=robust_desc)
            out[tid] = src_shared[tid]

    return main


@tilelang.jit(target="musa", out_idx=[1], pass_configs=PASS_CONFIGS)
def kernel_with_scalar_robust_force_async_copy_to_shared():

    @T.prim_func
    def main(
            src: T.Tensor([4], T.float32),
            out: T.Tensor([4], T.float32),
    ):
        with T.Kernel(1, threads=4) as _:
            tid = T.get_thread_binding()
            src_shared = T.alloc_shared([4], T.float32)
            robust_desc = T.make_robust_desc(T.address_of(src[1]), 8)
            T.copy(src[tid], src_shared[tid], force_async_copy=True, src_robust_desc=robust_desc)
            out[tid] = src_shared[tid]

    return main


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


@tilelang.jit(target="musa", out_idx=[1], pass_configs=PASS_CONFIGS)
def kernel_with_vectorized_scalar_robust_force_async_copy_to_shared():

    @T.prim_func
    def main(
            src: T.Tensor([4], T.float32),
            out: T.Tensor([4], T.float32),
    ):
        with T.Kernel(1, threads=1) as _:
            src_shared = T.alloc_shared([4], T.float32)
            robust_desc = T.make_robust_desc(T.address_of(src[1]), 8)
            for v in T.vectorized(4):
                T.copy(src[v], src_shared[v], force_async_copy=True, src_robust_desc=robust_desc)
            T.copy(src_shared, out)

    return main


@tilelang.jit(target="musa", out_idx=[1], pass_configs=PASS_CONFIGS_DISABLE_THREAD_STORAGE_SYNC)
def kernel_with_vectorized_scalar_robust_force_async_copy_to_shared_disable_thread_storage_sync():

    @T.prim_func
    def main(
            src: T.Tensor([4], T.float32),
            out: T.Tensor([4], T.float32),
    ):
        with T.Kernel(1, threads=1) as _:
            src_shared = T.alloc_shared([4], T.float32)
            robust_desc = T.make_robust_desc(T.address_of(src[1]), 8)
            for v in T.vectorized(4):
                T.copy(src[v], src_shared[v], force_async_copy=True, src_robust_desc=robust_desc)
            T.copy(src_shared, out)

    return main


@pytest.mark.parametrize(
    "kernel_builder",
    [
        kernel_with_robust_load,
        kernel_with_robust_async_copy,
        kernel_with_scalar_robust_copy_to_shared,
        kernel_with_scalar_robust_force_async_copy_to_shared,
        kernel_with_vectorized_scalar_robust_force_async_copy_to_shared,
    ],
)
def test_robust_copy_numerical(kernel_builder):
    require_musa()

    src = torch.tensor([1.0, -2.0, 3.0, -4.0], device="musa", dtype=torch.float32)
    expected = torch.tensor([0.0, -2.0, 3.0, 0.0], device="musa", dtype=torch.float32)

    kernel = kernel_builder()
    out = kernel(src)
    if isinstance(out, (tuple, list)):
        out = out[0]

    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)


def test_scalar_robust_copy_to_shared_get_tir():
    func = kernel_with_scalar_robust_copy_to_shared.get_tir()
    assert func is not None


def test_scalar_robust_force_async_copy_to_shared_source():
    code = kernel_with_scalar_robust_force_async_copy_to_shared().get_kernel_source()

    assert "tl::cp_async_gs_robust<4>" in code
    assert "tl::robust_load" not in code


def test_vectorized_scalar_force_async_copy_to_shared_source():
    code = kernel_with_vectorized_scalar_force_async_copy_to_shared().get_kernel_source()

    assert "tl::cp_async_gs<16>" in code
    assert "tl::cp_async_wait<0>();" in code


def test_vectorized_scalar_force_async_copy_to_shared_source_disable_thread_storage_sync():
    code = kernel_with_vectorized_scalar_force_async_copy_to_shared_disable_thread_storage_sync(
    ).get_kernel_source()

    assert "tl::cp_async_gs<16>" in code
    assert "tl::cp_async_wait<0>();" not in code


def test_vectorized_scalar_robust_force_async_copy_to_shared_source():
    code = kernel_with_vectorized_scalar_robust_force_async_copy_to_shared().get_kernel_source()

    assert "tl::cp_async_gs_robust<16>" in code
    assert "tl::cp_async_wait<0>();" in code
    assert "tl::robust_load" not in code


def test_vectorized_scalar_robust_force_async_copy_to_shared_source_disable_thread_storage_sync():
    code = kernel_with_vectorized_scalar_robust_force_async_copy_to_shared_disable_thread_storage_sync(
    ).get_kernel_source()

    assert "tl::cp_async_gs_robust<16>" in code
    assert "tl::cp_async_wait<0>();" not in code
    assert "tl::robust_load" not in code


def test_zero_sized_robust_async_copy_numerical():
    require_musa()

    src = torch.tensor([1.0, -2.0, 3.0, -4.0], device="musa", dtype=torch.float32)
    expected = torch.zeros([4], device="musa", dtype=torch.float32)

    kernel = kernel_with_zero_sized_robust_async_copy()
    out = kernel(src)
    if isinstance(out, (tuple, list)):
        out = out[0]

    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)
