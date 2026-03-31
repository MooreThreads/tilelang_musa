import tilelang
import torch
import tilelang.language as T

tilelang.disable_cache()


@tilelang.jit(target="musa", out_idx=[-1])
def reduce_sum(M, N, dtype="float32", threads=128):

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1, threads=threads) as _:
            A_local = T.alloc_fragment((M, N), dtype)
            B_local = T.alloc_fragment((M,), dtype)

            T.copy(A, A_local)
            T.reduce_sum(A_local, B_local, dim=1)
            T.copy(B_local, B)

    return main


M, N = 8, 256
threads = 256
kernel = reduce_sum(M, N, threads=threads)
print(kernel.get_kernel_source())

A = torch.randn(M, N, dtype=torch.float32, device="musa")
B = kernel(A)

B_ref = torch.sum(A, dim=1)

torch.testing.assert_close(B, B_ref, rtol=1e-2, atol=1e-2)

print("pass!")
