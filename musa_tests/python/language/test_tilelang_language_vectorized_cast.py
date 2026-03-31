import torch
import tilelang
import tilelang.testing
import tilelang.language as T

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
}

str2dtype = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8_e4m3": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


@tilelang.jit(target="musa", pass_configs=PASS_CONFIGS)
def vectorized_cast_kernel(M: int, dtype_A: str, dtype_B: str):
    assert M % 256 == 0

    @T.prim_func
    def main(
        A: T.Tensor[(M,), dtype_A],  # noqa: F821
        B: T.Tensor[(M,), dtype_B],  # noqa: F821
    ):
        with T.Kernel(1, threads=128):
            T.copy(A, B)

    return main


@tilelang.jit(target="musa", pass_configs=PASS_CONFIGS)
def parallel_vectorized_cast_kernel(M: int, dtype_A: str, dtype_B: str):
    assert M % 256 == 0

    @T.prim_func
    def main(
        A: T.Tensor[(M,), dtype_A],  # noqa: F821
        B: T.Tensor[(M,), dtype_B],  # noqa: F821
    ):
        with T.Kernel(1, threads=128):
            A_local = T.alloc_fragment((M,), dtype_A)
            B_local = T.alloc_fragment((M,), dtype_B)

            T.copy(A, A_local)
            for i in T.Parallel(M):
                B_local[i] = A_local[i]
            T.copy(B_local, B)

    return main


def make_input_tensor(M: int, dtype_str: str):
    if dtype_str.startswith("float8"):
        return torch.randn(M, dtype=torch.float32).musa().to(str2dtype[dtype_str])
    return torch.randn(M, dtype=str2dtype[dtype_str]).musa()


def run_vectorized_cast(src_dtype_str: str, dst_dtype_str: str, check_str: str, lanes: int = 2):
    """Run the vectorized cast kernel and check the correctness.
    Args:
        src_dtype_str: The source data type string.
        dst_dtype_str: The destination data type string.
        check_str: Used to ensure vectorized cast is used.
        lanes: The number of lanes of the source and destination data types.
    """

    M = 128 * lanes
    kernel = vectorized_cast_kernel(M, src_dtype_str, dst_dtype_str)
    kernel_parallel = parallel_vectorized_cast_kernel(M, src_dtype_str, dst_dtype_str)

    A = make_input_tensor(M, src_dtype_str)
    B = torch.zeros(M, dtype=str2dtype[dst_dtype_str]).musa()
    C = torch.zeros(M, dtype=str2dtype[dst_dtype_str]).musa()

    kernel(A, B)
    kernel_parallel(A, C)

    expected = A.to(str2dtype[dst_dtype_str])
    if dst_dtype_str.startswith("float8"):
        expected = expected.to(torch.float32)
        actual_b = B.to(torch.float32)
        actual_c = C.to(torch.float32)
    else:
        actual_b = B
        actual_c = C

    code = kernel.get_kernel_source()
    code_parallel = kernel_parallel.get_kernel_source()

    assert_kwargs = {}
    if src_dtype_str.startswith("float8") or dst_dtype_str.startswith("float8"):
        assert_kwargs = {"rtol": 0.0, "atol": 0.0}

    torch.testing.assert_close(actual_b, expected, **assert_kwargs)
    torch.testing.assert_close(actual_c, expected, **assert_kwargs)

    assert check_str in code and check_str in code_parallel, f"Cast {src_dtype_str} to {dst_dtype_str} with {lanes=} is not vectorized!"


def test_vectorized_cast():
    # fp8 -> fp16 / fp32
    run_vectorized_cast("float8_e4m3", "float16", "tl::cvt_fp8e4m3_to_half_x2", 2)
    run_vectorized_cast("float8_e4m3", "float16", "tl::cvt_fp8e4m3_to_half_x4", 4)
    run_vectorized_cast("float8_e4m3", "float16", "tl::cvt_fp8e4m3_to_half_x4", 8)
    run_vectorized_cast("float8_e5m2", "float16", "tl::cvt_fp8e5m2_to_half_x2", 2)
    run_vectorized_cast("float8_e5m2", "float16", "tl::cvt_fp8e5m2_to_half_x4", 4)
    run_vectorized_cast("float8_e5m2", "float16", "tl::cvt_fp8e5m2_to_half_x4", 8)
    run_vectorized_cast("float8_e4m3", "float32", "tl::cvt_fp8e4m3_to_float_x2", 2)
    run_vectorized_cast("float8_e4m3", "float32", "tl::cvt_fp8e4m3_to_float_x4", 4)
    run_vectorized_cast("float8_e4m3", "float32", "tl::cvt_fp8e4m3_to_float_x4", 8)
    run_vectorized_cast("float8_e5m2", "float32", "tl::cvt_fp8e5m2_to_float_x2", 2)
    run_vectorized_cast("float8_e5m2", "float32", "tl::cvt_fp8e5m2_to_float_x4", 4)
    run_vectorized_cast("float8_e5m2", "float32", "tl::cvt_fp8e5m2_to_float_x4", 8)

    # fp16 -> fp8 / fp32
    run_vectorized_cast("float16", "float8_e4m3", "tl::cvt_half_to_fp8e4m3_x2", 2)
    run_vectorized_cast("float16", "float8_e4m3", "tl::cvt_half_to_fp8e4m3_x4", 4)
    run_vectorized_cast("float16", "float8_e4m3", "tl::cvt_half_to_fp8e4m3_x4", 8)
    run_vectorized_cast("float16", "float8_e5m2", "tl::cvt_half_to_fp8e5m2_x2", 2)
    run_vectorized_cast("float16", "float8_e5m2", "tl::cvt_half_to_fp8e5m2_x4", 4)
    run_vectorized_cast("float16", "float8_e5m2", "tl::cvt_half_to_fp8e5m2_x4", 8)
    run_vectorized_cast("float16", "float32", "tl::cvt_half_to_float_x2", 2)
    run_vectorized_cast("float16", "float32", "tl::cvt_half_to_float_x4", 4)
    run_vectorized_cast("float16", "float32", "tl::cvt_half_to_float_x4", 8)

    # bf16 -> fp32
    run_vectorized_cast("bfloat16", "float32", "tl::cvt_bfloat16_to_float_x2", 2)
    run_vectorized_cast("bfloat16", "float32", "tl::cvt_bfloat16_to_float_x4", 4)
    run_vectorized_cast("bfloat16", "float32", "tl::cvt_bfloat16_to_float_x4", 8)

    # fp32 -> fp8 / fp16 / bf16
    run_vectorized_cast("float32", "float8_e4m3", "tl::cvt_float_to_fp8e4m3_x2", 2)
    run_vectorized_cast("float32", "float8_e4m3", "tl::cvt_float_to_fp8e4m3_x4", 4)
    run_vectorized_cast("float32", "float8_e4m3", "tl::cvt_float_to_fp8e4m3_x4", 8)
    run_vectorized_cast("float32", "float8_e5m2", "tl::cvt_float_to_fp8e5m2_x2", 2)
    run_vectorized_cast("float32", "float8_e5m2", "tl::cvt_float_to_fp8e5m2_x4", 4)
    run_vectorized_cast("float32", "float8_e5m2", "tl::cvt_float_to_fp8e5m2_x4", 8)
    run_vectorized_cast("float32", "float16", "tl::cvt_float_to_half_x2", 2)
    run_vectorized_cast("float32", "float16", "tl::cvt_float_to_half_x4", 4)
    run_vectorized_cast("float32", "float16", "tl::cvt_float_to_half_x4", 8)
    run_vectorized_cast("float32", "bfloat16", "tl::cvt_float_to_bfloat16_x2", 2)
    run_vectorized_cast("float32", "bfloat16", "tl::cvt_float_to_bfloat16_x4", 4)
    run_vectorized_cast("float32", "bfloat16", "tl::cvt_float_to_bfloat16_x4", 8)


if __name__ == "__main__":
    test_vectorized_cast()
    tilelang.testing.main()
