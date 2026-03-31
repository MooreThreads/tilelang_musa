import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit(out_idx=-1)
def get_var_assign_kernel():
    @T.prim_func
    def main(A: T.Tensor((2,), T.int32)):
        with T.Kernel(1) as _:
            a = T.alloc_var(T.int32, init=1)
            b = T.alloc_var(T.int32, init=a)
            a = 2
            d = T.alloc_var(T.int32, init=a)
            A[0] = b
            A[1] = d

    return main


# TODO: var init is not supported on hip.
@tilelang.testing.requires_musa
def test_var_assign() -> None:
    kernel = get_var_assign_kernel()
    res = kernel()
    assert res[0] == 1
    assert res[1] == 2


if __name__ == "__main__":
    tilelang.testing.main()
