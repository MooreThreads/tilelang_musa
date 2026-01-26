import torch
import pytest
import tilelang
import tilelang.language as T

tilelang.disable_cache()

def matmul(M, N, K,
           block_M, block_N, block_K,
           dtype="float16", accum_dtype="float"):
    @T.prim_func
    def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
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
    return type_map.get(elem_type, None)

def get_rtol(elem_type):
    rtol_map = {
        torch.float16: 1e-3,
        torch.bfloat16: 7.9e-3,
        torch.float8_e4m3fn: 1.25e-1,
    }
    return rtol_map.get(elem_type, None)


def get_atol(elem_type):
    atol_map = {
        torch.float16: 1e-3,
        torch.bfloat16: 1e-3,
        torch.float8_e4m3fn: 1.25e-1,
    }
    return atol_map.get(elem_type, None)

elem_type_list  = [torch.float16, torch.bfloat16, torch.float8_e4m3fn]
size_list       = [(1024, 1024, 1024)]
block_size_list = [(128, 128, 64), (32, 32, 32), (16, 16, 16),
                   (8, 8, 8), (4, 4, 4), (2, 2, 2)]
num_stages_list = [1]
test_params = [
    (elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    for elem_type in elem_type_list
    for (M, N, K) in size_list
    for (BLOCK_M, BLOCK_N, BLOCK_K) in block_size_list
    if M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0
]
@pytest.mark.parametrize("elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K", test_params)
def test_mm_kernel(elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    device = "musa"
    A = torch.randn((M, K), dtype=torch.float16, device=device).to(elem_type)
    B = torch.randn((K, N), dtype=torch.float16, device=device).to(elem_type)
    program = matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, dtype=get_tilelang_type(elem_type), accum_dtype="float32")
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_SQMMA: True,
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
    rtol = get_rtol(elem_type)
    atol = get_atol(elem_type)
    torch.testing.assert_close(ref_out.to(torch.float32), C.to(torch.float32),
                               rtol=rtol, atol=atol)
    print(f"elem_type={elem_type}, M={M}, N={N}, K={K}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
    print("Pass")
