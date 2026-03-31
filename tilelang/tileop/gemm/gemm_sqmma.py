from __future__ import annotations

from .gemm_base import GemmBase
from .inst import GemmInst
from tilelang.layout import make_sqmma_swizzled_layout, make_ph_sqmma_fragment_c
from tilelang.utils.language import is_shared, is_fragment
from tilelang import _ffi_api
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


class GemmSQMMA(GemmBase):
    @staticmethod
    def _as_const_bool(value, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, tir.IntImm):
            return bool(value.value)
        if hasattr(value, "value"):
            return bool(value.value)
        raise ValueError(f"{name} must be a constant Bool type, got {value}")

    @staticmethod
    def _as_const_int(value, name: str) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, tir.IntImm):
            return int(value.value)
        if hasattr(value, "value"):
            return int(value.value)
        raise ValueError(f"{name} must be a constant integer, got {value}")

    def _select_sqmma_inst_shape(self, block_size: int, target: Target) -> tuple[int, int, int]:
        inst_shape = _ffi_api.GemmPySelectSQMMAInstShape(self.gemm_node, int(block_size), target)
        if len(inst_shape) != 3:
            raise ValueError("SQMMA is selected but no valid SQMMA instruction shape is found")
        return int(inst_shape[0]), int(inst_shape[1]), int(inst_shape[2])

    def infer_layout(self, target: Target, thread_nums: int):
        if not self.is_gemm_ss():
            raise ValueError(f"Unsupported gemm combination for sqmma, A: {self.A.scope()}, B: {self.B.scope()}")

        warp_m, warp_n = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GemmInst.SQMMA)
        sqmma_inst = self._select_sqmma_inst_shape(int(thread_nums), target)

        c_bits = int(tvm.DataType(self.C.dtype).bits)
        c_layout = make_ph_sqmma_fragment_c(self.M, self.N, warp_m, warp_n, c_bits, sqmma_inst)

        a_shape = self.A.shape
        b_shape = self.B.shape
        if len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError("GemmSQMMA expects A/B to be at least 2D buffers")

        a_stride = int(a_shape[-2])
        a_continuous = int(a_shape[-1])
        a_layout = make_sqmma_swizzled_layout(self.A, continuity=a_continuous, k_major=not self.trans_A)
        if a_stride == 32 and a_continuous == 32:
            a_bits = int(tvm.DataType(self.A.dtype).bits)
            a_repeat_factor = (32 // a_bits) if 0 < a_bits <= 32 else 1
            a_layout = a_layout.repeat(1, a_repeat_factor)

        b_stride = int(b_shape[-2])
        b_continuous = int(b_shape[-1])
        b_layout_continuity = b_continuous if self.trans_B else sqmma_inst[1]
        b_layout = make_sqmma_swizzled_layout(self.B, continuity=b_layout_continuity, k_major=self.trans_B)
        if b_stride == 32 and b_continuous == 32:
            b_bits = int(tvm.DataType(self.B.dtype).bits)
            b_repeat_factor = (32 // b_bits) if 0 < b_bits <= 32 else 1
            b_layout = b_layout.repeat(1, b_repeat_factor)

        return {
            self.C: c_layout,
            self.A: a_layout,
            self.B: b_layout,
        }

    def _build_op_instance(self, warp_m: int, warp_n: int) -> str:
        op_name = ""
        if is_fragment(self.A):
            if not self.trans_A:
                op_name = "tl::gemm_rs"
            else:
                raise ValueError("gemm_rs requires the A operand to be in non-transposed layout")
        elif is_fragment(self.B):
            op_name = "tl::gemm_sr"
        else:
            op_name = "tl::gemm_ss"

        clear_accum_bool = self._as_const_bool(self.clear_accum, "clear_accum")
        stride_a = self._as_const_int(self.stride_A, "stride_A")
        stride_b = self._as_const_int(self.stride_B, "stride_B")
        offset_a = self._as_const_int(self.offset_A, "offset_A")
        offset_b = self._as_const_int(self.offset_B, "offset_B")
        wg_wait = self._as_const_int(self.wg_wait, "wg_wait")

        op_instance = (
            f"{op_name}<"
            f"{self.M}, {self.N}, {self.K}, "
            f"{warp_m}, {warp_n}, "
            f"{int(bool(self.trans_A))}, {int(bool(self.trans_B))}, {int(clear_accum_bool)}, "
            f"{stride_a}, {stride_b}, "
            f"{offset_a}, {offset_b}, "
            "true"
        )

        if wg_wait != 0:
            op_instance += f", {wg_wait}"
        op_instance += ">"
        return op_instance

    def lower(self, layout_map: dict, target: Target, thread_bounds: Range, thread_var: tir.Var):
        del layout_map, thread_var
        block_size = int(thread_bounds.extent)
        warp_m, warp_n = self.policy.compute_warp_partition(self.M, self.N, block_size, target, GemmInst.SQMMA)
        op_instance = self._build_op_instance(warp_m, warp_n)

        if not is_fragment(self.C):
            raise ValueError("GemmSQMMA only supports C in local.fragment scope")

        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        @T.prim_func
        def _gemm_sqmma() -> None:
            A_ptr = T.access_ptr(A_region, "r")
            B_ptr = T.access_ptr(B_region, "r")
            C_ptr = T.access_ptr(C_region, "rw")
            T.evaluate(
                T.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.tl_gemm"),
                    tir.StringImm(op_instance),
                    A_ptr,
                    B_ptr,
                    C_ptr,
                ))

        return _Simplify(_gemm_sqmma, inline_let=True)

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and is_fragment(self.B)

    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)
