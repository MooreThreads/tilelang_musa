import tilelang
import tilelang.language as T
import torch

TARGET = "musa"
DEVICE = "musa"
tilelang.disable_cache()


@tilelang.jit(target=TARGET)
def matmul(A, B, block_M=128, block_N=128, block_K=64, dtype="float16", accum_dtype="float", num_warp=4):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty((M, N), dtype)
    thread_per_block = num_warp * 32

    with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_per_block) as (
                bx,
                by,
            ):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local, wg_wait=-1)
            T.wait_wgmma()
        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def test_wait_wgmma():
    M, N, K = 512, 512, 512
    bm, bn, bk = 128, 128, 64
    dtype = "float16"
    acc_type = "float"
    kernel = matmul.compile(
        M=M,
        N=N,
        K=K,
        block_M=bm,
        block_N=bn,
        block_K=bk,
        dtype=dtype,
        accum_dtype=acc_type,
    )
    print(kernel.get_kernel_source())
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    c = kernel(a, b)
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@tilelang.jit(target=TARGET)
def matmul_with_independent_compute(A,
                                    B,
                                    block_M=128,
                                    block_N=128,
                                    block_K=64,
                                    dtype="float16",
                                    accum_dtype="float",
                                    num_warp=4):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty((M, N), dtype)
    D = T.empty((M, N), accum_dtype)
    thread_per_block = num_warp * 32

    with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_per_block) as (
                bx,
                by,
            ):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        D_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        T.clear(D_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local, wg_wait=-1)
            for i, j in T.Parallel(block_M, block_N):
                a_val = T.cast(A_shared[i, 0], accum_dtype)
                b_val = T.cast(B_shared[0, j], accum_dtype)
                D_local[i, j] += a_val + b_val
            T.wait_wgmma()
        T.copy(C_local, C[by * block_M, bx * block_N])
        T.copy(D_local, D[by * block_M, bx * block_N])

    return C, D


def independent_compute_reference(a, b, block_K):
    M, K = a.shape
    _, N = b.shape
    a_fp32 = a.to(torch.float32)
    b_fp32 = b.to(torch.float32)

    ref_d = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    for k_base in range(0, K, block_K):
        ref_d += a_fp32[:, k_base].unsqueeze(1)
        ref_d += b_fp32[k_base, :].unsqueeze(0)
    return ref_d


def test_wait_wgmma_with_independent_compute():
    M, N, K = 512, 512, 512
    bm, bn, bk = 128, 128, 64
    dtype = "float16"
    acc_type = "float"
    kernel = matmul_with_independent_compute.compile(
        M=M,
        N=N,
        K=K,
        block_M=bm,
        block_N=bn,
        block_K=bk,
        dtype=dtype,
        accum_dtype=acc_type,
    )

    print(kernel.get_kernel_source())
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    c, d = kernel(a, b)
    ref_c = a @ b
    ref_d = independent_compute_reference(a, b, bk)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(d, ref_d, rtol=1e-2, atol=1e-2)
