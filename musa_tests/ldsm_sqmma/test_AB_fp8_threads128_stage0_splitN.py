import torch
import pytest
import tilelang
import tilelang.language as T

tilelang.disable_cache()


def matmul(M,
           N,
           K,
           block_M,
           block_N,
           block_K,
           threads,
           stages,
           dtype="float16",
           accum_dtype="float"):

    @T.prim_func
    def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def get_tilelang_type(elem_type):
    type_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float8_e4m3fn: "float8_e4m3"
    }
    return type_map.get(elem_type)


elem_type_list = [torch.float8_e4m3fn]
size_list = [(4096, 4096, 4096)]
block_threads_stages_list = [
    (16, 128, 32, 128, 0),
    (16, 128, 64, 128, 0),
    (16, 128, 128, 128, 0),
    (16, 256, 32, 128, 0),
    (16, 256, 64, 128, 0),
    (16, 256, 128, 128, 0),
    (16, 512, 32, 128, 0),
    (16, 512, 64, 128, 0),
    (16, 512, 128, 128, 0),
    (32, 256, 32, 128, 0),
    (32, 256, 64, 128, 0),
    (32, 256, 128, 128, 0),
    (32, 512, 32, 128, 0),
    (32, 512, 64, 128, 0),
    (32, 512, 128, 128, 0),
    (64, 256, 32, 128, 0),
    (64, 256, 64, 128, 0),
    (64, 256, 128, 128, 0),
    (64, 512, 32, 128, 0),
    (64, 512, 64, 128, 0),
    (64, 512, 128, 128, 0),
    (128, 256, 32, 128, 0),
    (128, 256, 64, 128, 0),
    (128, 256, 128, 128, 0),
    (128, 512, 32, 128, 0),
    (128, 512, 64, 128, 0),
    (128, 512, 128, 128, 0),
]
test_params = [(elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, threads, stages)
               for elem_type in elem_type_list
               for (M, N, K) in size_list
               for (BLOCK_M, BLOCK_N, BLOCK_K, threads, stages) in block_threads_stages_list
               if M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0]


@pytest.mark.parametrize("elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, threads, stages",
                         test_params)
def test_mm_kernel(elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, threads, stages):
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
        threads,
        stages,
        dtype=get_tilelang_type(elem_type),
        accum_dtype="float32")
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True,
    }
    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target="musa",
        execution_backend="cython",
        verbose=True,
        pass_configs=pass_configs,
    )
    print(kernel.get_kernel_source())
    ref_out = A @ B
    C = kernel(A, B)
    torch.testing.assert_close(
        ref_out.to(torch.float16), C.to(torch.float16), rtol=1.25e-1, atol=1.25e-1)
    print(
        f"elem_type={elem_type}, M={M}, N={N}, K={K}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, threads={threads}"
    )
    print("Pass")
