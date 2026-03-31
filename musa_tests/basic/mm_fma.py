import tilelang
import tilelang.language as T

tilelang.disable_cache()

def matmul_fma(M,
                N,
                K,
                block_M,
                block_N,
                block_K,
                thread_tile_m=8,
                thread_tile_n=8,
                dtype="float16",
                accum_dtype="float"):
    threads_m = block_M // thread_tile_m
    threads_n = block_N // thread_tile_n
    assert block_M % thread_tile_m == 0 and block_N % thread_tile_n == 0, \
        "block size 必须能被 per-thread tile 整除"
    threads = threads_m * threads_n

    @T.prim_func
    def matmul_relu_kernel(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_local((thread_tile_m, thread_tile_n), accum_dtype)

            tid = T.get_thread_binding()
            lane_m = tid % threads_m
            lane_n = tid // threads_m

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                k_tile = T.min(block_K, K - ko * block_K)
                for kk in T.serial(k_tile):
                    for i in T.serial(thread_tile_m):
                        a_val = T.cast(
                            A_shared[lane_m * thread_tile_m + i, kk],
                            accum_dtype,
                        )
                        for j in T.serial(thread_tile_n):
                            b_val = T.cast(
                                B_shared[kk, lane_n * thread_tile_n + j],
                                accum_dtype,
                            )
                            C_local[i, j] += a_val * b_val

            zero = T.cast(0, accum_dtype)
            for i, j in T.grid(thread_tile_m, thread_tile_n):
                val = T.max(C_local[i, j], zero)
                row = by * block_M + lane_m * thread_tile_m + i
                col = bx * block_N + lane_n * thread_tile_n + j
                if row < M and col < N:
                    C[row, col] = val

    return matmul_relu_kernel


if __name__ == "__main__":
    M = 1024
    N = 1024
    K = 1024
    block_M = 128
    block_N = 128
    block_K = 32
    program = matmul_fma(M, N, K, block_M, block_N, block_K)
    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target="musa",
        execution_backend="cython",
        verbose=True,
    )
    # print("before lower")
    # program.show()
    # print("after lower")
    # kernel.artifact.device_mod.show()
    print(kernel.get_kernel_source())

    import torch

    # torch.musa.set_device(0)

    a = torch.randn(M, K, device="musa", dtype=torch.float16)
    b = torch.randn(K, N, device="musa", dtype=torch.float16)
    print('start kernel')
    c = kernel(a, b)
    print('start ref')
    ref_c = torch.relu(a @ b)
    print('compare')
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("fma matmul matches torch reference.")
