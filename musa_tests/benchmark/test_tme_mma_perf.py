import tilelang
import tilelang.language as T
import torch
import triton
import pytest
import tilelang.testing


# tilelang.disable_cache()
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def matmul_kernel(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):

        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=512) as (
                    bx,
                    by,
                ):
            T.use_swizzle(panel_size=4, order='col')
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, policy=T.GemmWarpPolicy.Square)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return matmul_kernel


def get_tilelang_type(elem_type):
    type_map = {
        torch.float16: "float16",
    }
    return type_map.get(elem_type)


def get_tflops(latency_ms, M, N, K):
    latency = latency_ms / 1e3
    tflops = (2.0 * M * N * K) / latency / 1e12
    return tflops


elem_type_list = [torch.float16]
# size_list       = [(16384, 16384, 16384), (8192, 8192, 8192), (4096, 4096, 4096), (2048, 2048, 2048), (1024, 1024, 1024)]
size_list = [(8192, 7168, 2048), (8192, 7168, 4096), (8192, 7168, 8192), (8192, 7168, 16384)]
block_size_list = [(256, 256, 64)]
num_stages_list = [1]
test_params = [(elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
               for elem_type in elem_type_list
               for (M, N, K) in size_list
               for (BLOCK_M, BLOCK_N, BLOCK_K) in block_size_list
               if M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0]


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K", test_params)
def test_mm_kernel_perf(elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    device = "musa"
    A = torch.randn((M, K), dtype=torch.float16, device=device).to(elem_type)
    B = torch.randn((K, N), dtype=torch.float16, device=device).to(elem_type)
    program = matmul(
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        dtype=get_tilelang_type(elem_type),
        accum_dtype="float32")

    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }
    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target="musa",
        execution_backend="cython",
        verbose=True,
        pass_configs=pass_configs,
    )
    ms_torch = triton.musa_testing.do_bench(lambda: torch.mm(A, B), warmup=25, rep=100)
    ms_tilelang = triton.musa_testing.do_bench(lambda: kernel(A, B), warmup=25, rep=100)
    print("\n")
    print(
        f"elem_type={elem_type}, M={M}, N={N}, K={K}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}"
    )
    print(f"torch latency: {ms_torch:.4f} ms")
    print(f"tilelang kernel latency: {ms_tilelang:.4f} ms")
    print(f"torch tflops: {get_tflops(ms_torch, M, N, K):.4f}")
    print(f"tilelang kernel tflops: {get_tflops(ms_tilelang, M, N, K):.4f}")
    print(f"tilelang2torch speed_up: {round(ms_torch / ms_tilelang, 6)}")
    ref_out = torch.mm(A, B)
    C = kernel(A, B)
    torch.testing.assert_close(
        ref_out.to(torch.float16), C.to(torch.float16), rtol=1.25e-1, atol=1.25e-1)
