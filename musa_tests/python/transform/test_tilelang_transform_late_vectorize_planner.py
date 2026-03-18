import torch
import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
}


@tilelang.jit(target="musa", out_idx=[1], pass_configs=PASS_CONFIGS)
def late_vectorize_planner_regression():

    @T.prim_func
    def main(
            A: T.Tensor((32,), "float32"),
            B: T.Tensor((32,), "float32"),
    ):
        with T.Kernel(1, threads=1):
            for i in T.serial(4):
                for j in T.vectorized(8):
                    B[i * 8 + j] = T.exp2(A[i * 8 + j])

    return main


def test_late_vectorize_planner_end_to_end():
    kernel = late_vectorize_planner_regression()
    inp = torch.linspace(-2.0, 2.0, 32, device="musa", dtype=torch.float32)
    out = kernel(inp)
    if isinstance(out, (tuple, list)):
        out = out[0]
    expected = torch.exp2(inp)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    tilelang.testing.main()
