from tilelang import tvm as tvm
import tilelang as tl
from tilelang.utils.target import determine_target
import tilelang.language as T
import tilelang.testing
from tvm import tir

tilelang.disable_cache()

auto_target = tvm.target.Target(determine_target("auto"))
print(f"Running tests on target: {auto_target}")

def _get_outer_loop_kind(stmt) -> tir.ForKind:
    outer_kind = None

    def visit(node):
        nonlocal outer_kind
        if isinstance(node, tir.For) and isinstance(node.body, tir.For):
            if node.body.kind == tir.ForKind.VECTORIZED:
                outer_kind = node.kind

    tir.stmt_functor.post_order_visit(stmt, visit)

    assert outer_kind is not None, "Failed to find the vectorize-split outer loop"
    return outer_kind


def _legalize_vectorized_loop(disable_auto_unroll: bool) -> tir.ForKind:

    @T.prim_func
    def before(A: T.Tensor((64,), "float16"), B: T.Tensor((64,), "float16")):
        with T.block("root"):
            for i in T.vectorized(64):
                B[i] = A[i]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)

    config = {"tl.disable_auto_unroll": disable_auto_unroll}
    with tvm.transform.PassContext(config=config):
        with auto_target:
            mod = tl.transform.LegalizeVectorizedLoop()(mod)

    return _get_outer_loop_kind(mod["main"].body)


def test_vectorize_auto_unroll_on():
    # Default: TL_DISABLE_AUTO_UNROLL = False (auto unroll enabled)
    assert _legalize_vectorized_loop(False) == tir.ForKind.UNROLLED


def test_vectorize_auto_unroll_disable():
    assert _legalize_vectorized_loop(True) == tir.ForKind.SERIAL


if __name__ == "__main__":
    tilelang.testing.main()
