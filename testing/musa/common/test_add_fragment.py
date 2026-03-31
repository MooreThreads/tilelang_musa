import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()


@tilelang.jit(target="musa", verbose=True)
def elementwise_add(A, B, num_per_thread=8, threads=256, dtype="float32"):
    N = T.const("N")
    A: T.Tensor[[N], dtype]
    B: T.Tensor[[N], dtype]
    C = T.empty((N,), dtype)
    num_per_block = threads * num_per_thread

    with T.Kernel(T.ceildiv(N, num_per_block), threads=threads) as b_x:
        sum_local = T.alloc_fragment((num_per_block,), dtype)
        # vector add through local fragment.
        for i, j in T.Parallel(threads, num_per_thread):
            offsets = i * num_per_thread + j
            sum_local[offsets] = A[b_x * num_per_block + offsets] + B[b_x * num_per_block + offsets]
        T.copy(sum_local, C[b_x * num_per_block])

    return C


def ref_program(x, y):
    return x + y


def test_elementwise_add():
    N = 4096
    kernel = elementwise_add.compile(N=N)
    a = torch.randn(N, dtype=torch.float32, device="musa")
    b = torch.randn(N, dtype=torch.float32, device="musa")
    c = kernel(a, b)
    torch.testing.assert_close(c, ref_program(a, b), rtol=1e-2, atol=1e-2)


def main():
    N = 4096
    kernel = elementwise_add.compile(N=N)
    print(kernel.get_kernel_source())

    a = torch.randn(N, dtype=torch.float32, device="musa")
    b = torch.randn(N, dtype=torch.float32, device="musa")

    c = kernel(a, b)
    torch.testing.assert_close(c, ref_program(a, b), rtol=1e-2, atol=1e-2)
    print("pass!")


if __name__ == "__main__":
    main()
