import tilelang
import tilelang.language as T
import torch
from tvm import tir


@tilelang.jit
def kernel_with_warp_sync():

    @T.prim_func
    def main(
            A: T.Tensor((1,), "int32"),
            B: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if tx == 0:
                tir.call_extern("void", "__nanosleep", 100)
                A[0] = -1
            T.sync_warp()
            if tx == 1:
                B[0] = A[0]

    return main


def test_warp_sync():
    a = torch.empty((1), device="musa", dtype=torch.int32)
    b = torch.empty((1), device="musa", dtype=torch.int32)
    kernel = kernel_with_warp_sync()
    print(kernel.get_kernel_source())
    assert "__syncwarp" in kernel.get_kernel_source()
    kernel(a, b)
    assert b[0] == -1
