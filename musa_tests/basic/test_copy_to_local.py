import pytest
import torch
import tilelang
import tilelang.language as T

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(target="musa", out_idx=[1], pass_configs=PASS_CONFIGS)
def kernel_with_copy():

    @T.prim_func
    def main(
            quant_scales: T.Tensor([4], T.float32),
            out: T.Tensor([256], T.float32),
    ):
        with T.Kernel(1, threads=256) as _:
            quant_local = T.alloc_local([4], T.float32)
            T.copy(quant_scales, quant_local)
            for i in T.Parallel(256):
                out[i] = quant_local[i // 64]

    return main


@tilelang.jit(target="musa", out_idx=[1], pass_configs=PASS_CONFIGS)
def kernel_with_parallel_store():

    @T.prim_func
    def main(
            quant_scales: T.Tensor([4], T.float32),
            out: T.Tensor([256], T.float32),
    ):
        with T.Kernel(1, threads=256) as _:
            quant_local = T.alloc_local([4], T.float32)
            for i in T.Parallel(4):
                quant_local[i] = quant_scales[i]
            for i in T.Parallel(256):
                out[i] = quant_local[i // 64]

    return main


def reference_output(quant_scales: torch.Tensor) -> torch.Tensor:
    return quant_scales.repeat_interleave(64)


def test_copy_to_local_codegen():
    kernel = kernel_with_copy()
    source = kernel.get_kernel_source()

    # Buggy pattern: thread-partitioned writes to a thread-local buffer.
    bad_if = "if (((int)threadIdx.x) < 4)"
    bad_assign = "quant_local[((int)threadIdx.x)] = quant_scales[((int)threadIdx.x)]"
    assert not (bad_if in source and bad_assign
                in source), "T.copy to local buffer is still lowered to thread-partitioned writes."


@pytest.mark.parametrize(
    "kernel_builder",
    [kernel_with_copy, kernel_with_parallel_store],
)
def test_copy_to_local_numerical(kernel_builder):
    quant_scales = torch.tensor([1.0, -2.0, 3.0, -4.0], device="musa", dtype=torch.float32)
    kernel = kernel_builder()
    out = kernel(quant_scales)
    if isinstance(out, (tuple, list)):
        out = out[0]
    ref = reference_output(quant_scales)
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)
