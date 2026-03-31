import contextlib
import sys
import inspect
import pytest
import random
import torch
import numpy as np
from tilelang.contrib import mcc, nvcc
from tvm.testing.utils import (
    requires_cuda,
    requires_package,
    requires_llvm,
    requires_metal,
    requires_rocm,
    _compose,
)

from tilelang.utils.tensor import torch_assert_close as torch_assert_close
from .perf_regression import process_func, regression

__all__ = (
    [
        "requires_package",
        "requires_cuda",
        "requires_musa",
        "requires_metal",
        "requires_rocm",
        "requires_llvm",
        "main",
        "matmul_naive",
        "requires_cuda_compute_version",
        "requires_musa_compute_version",
        "process_func",
        "regression",
    ]
    + [f"requires_cuda_compute_version_{op}" for op in ("ge", "gt", "le", "lt", "eq")]
    + [f"requires_musa_compute_version_{op}" for op in ("ge", "gt", "le", "lt", "eq")]
)


def matmul_naive(A, B, accum_dtype, out_dtype):
    """Reference matmul with explicit outer-product accumulation."""
    C = torch.zeros((A.shape[0], B.shape[1]), dtype=accum_dtype, device=A.device)
    for k in range(A.shape[1]):
        C.addcmul_(A[:, k : k + 1].to(accum_dtype), B[k : k + 1, :].to(accum_dtype))
    return C.to(out_dtype)


# pytest.main() wrapper to allow running single test file
def main():
    test_file = inspect.getsourcefile(sys._getframe(1))
    sys.exit(pytest.main([test_file] + sys.argv[1:]))


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with contextlib.suppress(BaseException):
        torch.musa.manual_seed_all(seed)


def requires_musa(func=None):
    """Mark a test as requiring MUSA support."""
    try:
        mcc.find_musa_path()
        has_musa = True
        reason = ""
    except RuntimeError as err:
        has_musa = False
        reason = f"MUSA is not available: {err}"

    requires = [pytest.mark.skipif(not has_musa, reason=reason)]

    def inner(func):
        return _compose([func], requires)

    if func is None:
        return inner
    return inner(func)


def requires_cuda_compute_version(major_version, minor_version=0, mode="ge"):
    """Mark a test as requiring at least a compute architecture

    Unit test marked with this decorator will run only if the CUDA
    compute architecture of the GPU is at least `(major_version,
    minor_version)`.

    This also marks the test as requiring a cuda support.

    Parameters
    ----------
    major_version: int

        The major version of the (major,minor) version tuple.

    minor_version: int

        The minor version of the (major,minor) version tuple.

    mode: str

        The mode of the comparison.
        - "ge": greater than or equal to
        - "gt": greater than
        - "le": less than or equal to
        - "lt": less than
    """
    min_version = (major_version, minor_version)
    try:
        arch = nvcc.get_target_compute_version()
        compute_version = nvcc.parse_compute_version(arch)
    except ValueError:
        # No GPU present.  This test will be skipped from the
        # requires_cuda() marks as well.
        compute_version = (0, 0)

    min_version_str = ".".join(str(v) for v in min_version)
    compute_version_str = ".".join(str(v) for v in compute_version)

    def compare(compute_version, min_version, mode) -> bool:
        if mode == "ge":
            return compute_version >= min_version
        elif mode == "gt":
            return compute_version > min_version
        elif mode == "le":
            return compute_version <= min_version
        elif mode == "lt":
            return compute_version < min_version
        elif mode == "eq":
            return compute_version == min_version
        else:
            raise ValueError(f"Invalid mode: {mode}")

    requires = [
        pytest.mark.skipif(
            not compare(compute_version, min_version, mode),
            reason=f"Requires CUDA compute {mode} {min_version_str}, but have {compute_version_str}",
        ),
        *requires_cuda.marks(),
    ]

    def inner(func):
        return _compose([func], requires)

    return inner


def requires_cuda_compute_version_ge(major_version, minor_version=0):
    return requires_cuda_compute_version(major_version, minor_version, mode="ge")


def requires_cuda_compute_version_gt(major_version, minor_version=0):
    return requires_cuda_compute_version(major_version, minor_version, mode="gt")


def requires_cuda_compute_version_eq(major_version, minor_version=0):
    return requires_cuda_compute_version(major_version, minor_version, mode="eq")


def requires_cuda_compute_version_lt(major_version, minor_version=0):
    return requires_cuda_compute_version(major_version, minor_version, mode="lt")


def requires_cuda_compute_version_le(major_version, minor_version=0):
    return requires_cuda_compute_version(major_version, minor_version, mode="le")





def requires_musa_compute_version(major_version, minor_version=0, mode="ge"):
    """Mark a test as requiring at least a compute architecture

    Unit test marked with this decorator will run only if the MUSA
    compute architecture of the GPU is at least `(major_version,
    minor_version)`.

    This also marks the test as requiring a musa support.

    Parameters
    ----------
    major_version: int

        The major version of the (major,minor) version tuple.

    minor_version: int

        The minor version of the (major,minor) version tuple.

    mode: str

        The mode of the comparison.
        - "ge": greater than or equal to
        - "gt": greater than
        - "le": less than or equal to
        - "lt": less than
    """
    min_version = (major_version, minor_version)
    try:
        arch = mcc.get_musa_compute_version()
        compute_version = mcc.parse_musa_compute_version(arch)
    except ValueError:
        # No GPU present.  This test will be skipped from the
        # requires_musa() marks as well.
        compute_version = (0, 0)

    min_version_str = ".".join(str(v) for v in min_version)
    compute_version_str = ".".join(str(v) for v in compute_version)

    def compare(compute_version, min_version, mode) -> bool:
        if mode == "ge":
            return compute_version >= min_version
        elif mode == "gt":
            return compute_version > min_version
        elif mode == "le":
            return compute_version <= min_version
        elif mode == "lt":
            return compute_version < min_version
        elif mode == "eq":
            return compute_version == min_version
        else:
            raise ValueError(f"Invalid mode: {mode}")

    requires = [
        pytest.mark.skipif(
            not compare(compute_version, min_version, mode),
            reason=f"Requires MUSA compute {mode} {min_version_str}, but have {compute_version_str}",
        ),
    ]

    def inner(func):
        return _compose([func], requires)

    return inner


def requires_musa_compute_version_ge(major_version, minor_version=0):
    return requires_musa_compute_version(major_version, minor_version, mode="ge")


def requires_musa_compute_version_gt(major_version, minor_version=0):
    return requires_musa_compute_version(major_version, minor_version, mode="gt")


def requires_musa_compute_version_eq(major_version, minor_version=0):
    return requires_musa_compute_version(major_version, minor_version, mode="eq")


def requires_musa_compute_version_lt(major_version, minor_version=0):
    return requires_musa_compute_version(major_version, minor_version, mode="lt")


def requires_musa_compute_version_le(major_version, minor_version=0):
    return requires_musa_compute_version(major_version, minor_version, mode="le")
