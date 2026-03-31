import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()


@tilelang.jit(target="musa", verbose=True)
def add_one_with_unroll_factor(A, num_per_thread=8, threads=256, dtype="float32"):
    N = T.const("N")
    A: T.Tensor[[N], dtype]
    B = T.empty((N,), dtype)

    with T.Kernel(T.ceildiv(N, threads * num_per_thread), threads=threads) as b_x:
        for i in T.Parallel(threads):
            base = (b_x * threads + i) * num_per_thread
            for j in T.unroll(0, num_per_thread, unroll_factor=4):
                B[base + j] = A[base + j] + 1.0

    return B


def ref_program(x):
    return x + 1.0


def test_unroll_factor_codegen_and_numerical():
    N = 4096
    kernel = add_one_with_unroll_factor.compile(N=N)
    source = kernel.get_kernel_source()
    assert "#pragma unroll 4" in source

    a = torch.randn(N, dtype=torch.float32, device="musa")
    b = kernel(a)
    torch.testing.assert_close(b, ref_program(a), rtol=1e-2, atol=1e-2)


def main():
    N = 4096
    kernel = add_one_with_unroll_factor.compile(N=N)
    print(kernel.get_kernel_source())

    a = torch.randn(N, dtype=torch.float32, device="musa")
    b = kernel(a)
    torch.testing.assert_close(b, ref_program(a), rtol=1e-2, atol=1e-2)
    print("pass!")


if __name__ == "__main__":
    main()
