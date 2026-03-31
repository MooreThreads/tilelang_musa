from .gemm_base import GemmBase
from .inst import GemmInst
from tilelang.layout import make_gemm_fragment_c_linear, make_gemm_layout_linear
from tilelang.utils.language import is_shared, is_fragment
from tvm.target import Target
from tvm.ir import Range
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


class GemmFMA(GemmBase):
    def infer_layout(self, target: Target, thread_nums: int):
        # Keep the warp partition query in sync with C++ GemmWarpPolicy.
        self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GemmInst.FMA)

        if not self.is_gemm_ss():
            raise ValueError(f"Unsupported gemm combination for fma, A: {self.A.scope()}, B: {self.B.scope()}")

        a_shape = self.A.shape
        b_shape = self.B.shape
        if len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError("GemmFMA expects A/B to be at least 2D buffers")

        a_stride = int(a_shape[-2])
        a_continuous = int(a_shape[-1])
        b_stride = int(b_shape[-2])
        b_continuous = int(b_shape[-1])

        return {
            self.C: make_gemm_fragment_c_linear(self.M, self.N, thread_nums),
            self.A: make_gemm_layout_linear(a_stride, a_continuous),
            self.B: make_gemm_layout_linear(b_stride, b_continuous),
        }

    def lower(self, layout_map: dict, target: Target, thread_bounds: Range, thread_var: tir.Var):
        del layout_map  # FMA lowering does not need layout remap input.
        thread_nums = thread_bounds.extent
        accum_dtype = self.accum_dtype
        clear_accum = self.clear_accum

        A_buf = self.A
        B_buf = self.B
        C_buf = self.C

        M = self.M
        N = self.N
        K = self.K
        trans_A = self.trans_A
        trans_B = self.trans_B

        if not self.is_gemm_ss():
            raise ValueError(f"Unsupported gemm combination for fma, A: {self.A.scope()}, B: {self.B.scope()}")

        @T.prim_func
        def _gemm_fma() -> None:
            accum = T.alloc_local((1,), accum_dtype)
            total = M * N
            trip = T.ceildiv(total, thread_nums)
            for idx_iter in T.serial(0, trip):
                linear = idx_iter * thread_nums + thread_var
                if linear < total:
                    i = linear // N
                    j = linear - i * N

                    if clear_accum:
                        accum[0] = T.cast(0, accum_dtype)
                    else:
                        accum[0] = C_buf[i, j]

                    for k_iter in T.serial(0, K):
                        if trans_A:
                            a_val = T.cast(A_buf[k_iter, i], accum_dtype)
                        else:
                            a_val = T.cast(A_buf[i, k_iter], accum_dtype)
                        if trans_B:
                            b_val = T.cast(B_buf[j, k_iter], accum_dtype)
                        else:
                            b_val = T.cast(B_buf[k_iter, j], accum_dtype)
                        accum[0] = accum[0] + a_val * b_val
                    C_buf[i, j] = accum[0]

        return _Simplify(_gemm_fma, inline_let=True)

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and is_fragment(self.B)

    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)
