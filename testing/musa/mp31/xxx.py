
@tilelang.testing.requires_musa
def test_sync_hoist_non_uniform_if_non_warp_multiple_participants():
    """Sync inside non-uniform if must never be silently dropped.

    This regression covers the `thread_count % 32 != 0` surface (e.g. `tx < 50`)
    where partial-sync lowering cannot keep a partial barrier and therefore
    ThreadSync must conservatively hoist to a full-block sync.
    """

    @T.prim_func(private=True)
    def func():
        temp_shared = T.alloc_buffer([128], dtype="float32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        temp_shared[tx] = T.float32(tx)
        if tx < 50:
            result_local[0] = temp_shared[(tx + 1) % 128]

    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.ThreadSync("shared")(mod)
    s = str(mod)
    assert 'T.tvm_storage_sync("shared")' in s, f"Expected sync:\n{s}"
    sync_pos = s.index('T.tvm_storage_sync("shared")')
    if_pos = s.index("if tx < 50")
    assert sync_pos < if_pos, f"Sync should be hoisted before non-uniform if:\n{s}"

