import tilelang
import tilelang.language as T
from tilelang import tvm
from tvm import tir

tilelang.disable_cache()


def run_lower_shared_barrier(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(
        tvm.target.Target("musa", host="llvm")
    )(mod)
    mod = tilelang.transform.LowerSharedBarrier()(mod)
    return mod["main"]


def make_before():
    @T.prim_func
    def before():
        with T.Kernel(1, threads=128):
            mbars = T.alloc_buffer((2,), "uint64", scope="shared.barrier")
            T.block_attr({"barrier_init": {mbars.data: [T.int32(128), T.int32(256)]}})
            T.evaluate(tir.Call("handle", "tir.ptx_arrive_barrier", [mbars[0]]))
            T.evaluate(
                tir.Call(
                    "handle",
                    "tir.ptx_arrive_barrier_expect_tx",
                    [mbars[1], T.int32(64)],
                )
            )

    return before


def has_shared_barrier_alloc(stmt):
    has_alloc = False

    def visit(node):
        nonlocal has_alloc
        if isinstance(node, tir.Block):
            for buf in node.alloc_buffers:
                if getattr(buf.data.type_annotation, "storage_scope", None) == "shared.barrier":
                    has_alloc = True

    tir.stmt_functor.post_order_visit(stmt, visit)
    return has_alloc


def collect_lowered_stats(stmt):
    placeholder_calls = 0
    barrier_arg_is_buffer_load = False

    def visit(node):
        nonlocal placeholder_calls, barrier_arg_is_buffer_load
        if isinstance(node, tir.Call):
            op = node.op
            if isinstance(op, tvm.ir.Op) and op.name == "tl.barrier_id_placeholder":
                placeholder_calls += 1
            if isinstance(op, tvm.ir.Op) and op.name in {
                "tir.ptx_init_barrier_thread_count",
                "tir.ptx_arrive_barrier",
                "tir.ptx_arrive_barrier_expect_tx",
            }:
                if node.args and isinstance(node.args[0], tir.BufferLoad):
                    barrier_arg_is_buffer_load = True

    tir.stmt_functor.post_order_visit(stmt, visit)
    return placeholder_calls, barrier_arg_is_buffer_load


def test_lower_shared_barrier_to_named_barrier():
    lowered = run_lower_shared_barrier(make_before())

    assert not has_shared_barrier_alloc(lowered.body)
    placeholder_calls, barrier_arg_is_buffer_load = collect_lowered_stats(lowered.body)
    assert placeholder_calls == 2
    assert not barrier_arg_is_buffer_load


def main():
    before = make_before()
    print("=== Before LowerSharedBarrier ===")
    print(before.script())

    lowered = run_lower_shared_barrier(before)
    print("=== After LowerSharedBarrier ===")
    print(lowered.script())

    assert not has_shared_barrier_alloc(lowered.body)
    placeholder_calls, barrier_arg_is_buffer_load = collect_lowered_stats(lowered.body)
    assert placeholder_calls == 2
    assert not barrier_arg_is_buffer_load
    print("pass!")


if __name__ == "__main__":
    main()
