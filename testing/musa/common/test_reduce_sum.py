import tilelang
import torch
import tilelang.language as T

tilelang.disable_cache()


@tilelang.jit(target="musa", verbose=True)
def reduce_sum(A, dtype="float32", threads=256):
    M, N = T.const("M N")
    A: T.Tensor[[M, N], dtype]
    B = T.empty((M,), dtype)

    with T.Kernel(1, threads=threads) as _:
        A_local = T.alloc_fragment((M, N), dtype)
        B_local = T.alloc_fragment((M,), dtype)

        T.copy(A, A_local)
        T.reduce_sum(A_local, B_local, dim=1)
        T.copy(B_local, B)

    return B


def ref_program(x):
    return torch.sum(x, dim=1)


def test_reduce_sum():
    M = 8
    N = 256
    threads = 256
    kernel = reduce_sum.compile(M=M, N=N, threads=threads)
    a = torch.randn(M, N, dtype=torch.float32, device="musa")
    b = kernel(a)
    torch.testing.assert_close(b, ref_program(a), rtol=1e-2, atol=1e-2)


def main():
    M = 8
    N = 256
    threads = 256
    kernel = reduce_sum.compile(M=M, N=N, threads=threads)
    print(kernel.get_kernel_source())

    a = torch.randn(M, N, dtype=torch.float32, device="musa")
    b = kernel(a)

    torch.testing.assert_close(b, ref_program(a), rtol=1e-2, atol=1e-2)
    print("pass!")


if __name__ == "__main__":
    main()
