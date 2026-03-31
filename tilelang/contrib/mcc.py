# pylint: disable=invalid-name
# modified from apache tvm python/tvm/contrib/mcc.py
"""Utility to invoke mcc compiler in the system"""

from __future__ import annotations

import os
import subprocess
import warnings
import contextlib
from tilelang.env import MUSA_HOME, MUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH, env
import shutil
import tempfile
import tvm_ffi
from tilelang import tvm as tvm
from tvm.target import Target

from tvm.base import py_str
from tvm.contrib import utils


def compile_musa(
    code, target_format="ptx", arch=None, options=None, path_target=None, verbose=False
):
    """Compile musa code with MCC from env.

    Parameters
    ----------
    code : str
        The musa code.

    target_format : str
        The target format of mcc compiler.

    arch : str
        The musa architecture.

    options : str or list of str
        The additional options.

    path_target : str, optional
        Output file.

    Return
    ------
    mubin : bytearray
        The bytearray of the mubin
    """
    if arch is None:
        # If None, then it will use `tvm.target.Target.current().arch`.
        # Target arch could be a str like "mp_xx"
        compute_version = get_musa_compute_version(Target.current(allow_none=True))
        target_arch = get_musa_arch(compute_version)
        arch = f"mp_{target_arch}"

    temp = utils.tempdir(keep_for_debug=not env.should_cleanup_temp_files())
    file_name = "tvm_kernels"
    if target_format not in ["mubin", "asm"]:
        raise ValueError("target_format must be in mubin, asm")
    temp_code = temp.relpath(f"{file_name}.mu")
    temp_target = temp.relpath(f"{file_name}.{target_format}")

    pass_context = tvm.get_global_func("transform.GetCurrentPassContext")()
    kernels_output_dir = pass_context.config.get("musa.kernels_output_dir", None)
    if kernels_output_dir is not None:
        if not os.path.isdir(kernels_output_dir):
            os.makedirs(kernels_output_dir)
        temp_code = os.path.join(kernels_output_dir, f"{file_name}.mu")
        temp_target = os.path.join(kernels_output_dir, f"{file_name}.{target_format}")

    with open(temp_code, "w") as out_file:
        out_file.write(code)

    file_target = path_target if path_target else temp_target
    cmd = [get_mcc_compiler(), "--cuda-device-only"]
    if target_format == "asm":
        cmd += ["-S"]
    # TODO(xtyi): check lineinfo
    cmd += [f"--cuda-gpu-arch={arch}"]

    if options:
        if isinstance(options, str):
            cmd += [options]
        elif isinstance(options, list):
            cmd += options
        else:
            raise ValueError("options must be str or list of str")

    cmd += ["-o", file_target]
    cmd += [temp_code]

    # NOTE: ccbin option can be used to tell mcc where to find the c++ compiler
    # just in case it is not in the path. On Windows it is not in the path by default.
    # However, we cannot use TVM_CXX_COMPILER_PATH because the runtime env.
    # Because it is hard to do runtime compiler detection, we require mcc is configured
    # correctly by default.
    # if cxx_compiler_path != "":
    #    cmd += ["-ccbin", cxx_compiler_path]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()

    if verbose:
        print(py_str(out))

    if proc.returncode != 0:
        msg = (
            f"{code}\n"
            f"Compilation error:\n"
            f"{py_str(out)}\n"
            f"Command: {' '.join(cmd)}\n"
        )
        raise RuntimeError(msg)

    with open(file_target, "rb") as f:
        data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        return data


def default_compile_options(compile_flags: list[str] | None = None) -> list[str]:
    """
    Build a set of default MCC compile options for TileLang generated sources.

    Includes C++ standard and common include paths (TileLang templates, MUTLASS,
    MUSA include). Merges user-provided compile flags if given.

    Parameters
    ----------
    compile_flags : Optional[List[str]]
        Additional flags to include. Items are split on whitespace.

    Returns
    -------
    List[str]
        A list of flags suitable for MCC's command line.
    """
    options: list[str] = ["-std=c++17"]
    try:
        if TILELANG_TEMPLATE_PATH:
            options.append(f"-I{TILELANG_TEMPLATE_PATH}")
    except Exception:
        pass

    try:
        if MUTLASS_INCLUDE_DIR:
            options.append(f"-I{MUTLASS_INCLUDE_DIR}")
    except Exception:
        pass

    try:
        if MUSA_HOME:
            options.append(f"-I{os.path.join(MUSA_HOME, 'include')}")
    except Exception:
        pass

    # Preserve user flags exactly, including repeated tokens required by MCC
    # (e.g., multiple "-gencode" pairs or repeated "-Xcompiler" entries).
    if compile_flags:
        import shlex

        for flag in compile_flags:
            # Split each string like a shell would, preserving quoted args
            tokens = shlex.split(flag) if isinstance(flag, str) else [str(flag)]
            options.extend(tokens)
    return options


def find_musa_path():
    """Utility function to find musa path

    Returns
    -------
    path : str
        Path to musa root.
    """
    if MUSA_HOME:
        return MUSA_HOME
    raise RuntimeError(
        "Failed to automatically detect MUSA installation. Please set the MUSA_HOME environment variable manually (e.g., export MUSA_HOME=/usr/local/musa)."
    )


def get_musa_version(musa_path=None):
    """Utility function to get musa version

    Parameters
    ----------
    musa_path : Optional[str]

        Path to musa root.  If None is passed, will use
        `find_musa_path()` as default.

    Returns
    -------
    version : float
        The musa version

    """
    if musa_path is None:
        musa_path = find_musa_path()

    # todo: read from version.json

    cmd = [os.path.join(musa_path, "bin", "mcc"), "--version"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    out = py_str(out)
    if proc.returncode == 0:
        release_line = [l for l in out.split("\n") if "mcc version" in l][0]
        release_fields = [s.strip() for s in release_line.split(" ")]
        version_str = release_fields[-1]
        return tuple(int(field) for field in version_str.split("."))
    raise RuntimeError("Cannot read musa version file")


@tvm_ffi.register_global_func("tvm.contrib.mcc.get_compute_version", override=True)
def get_musa_compute_version(target=None):
    """Utility function to get compute capability of compilation target.

    Looks for the target arch in three different places, first in the target input, then the
    Target.current() scope, and finally the GPU device (if it exists).

    Parameters
    ----------
    target : tvm.target.Target, optional
        The compilation target

    Returns
    -------
    compute_version : str
        compute capability of a GPU (e.g. "8.6" or "9.0")
    """
    # 1. input target object
    # 2. Target.current()
    target = target or Target.current()
    if target and target.arch:
        arch = str(target.arch).split("_")[-1]
        if "." in arch:
            return arch
        if len(arch) < 2:
            raise ValueError(f"Invalid MUSA arch format: {target.arch}")
        if len(arch) == 2:
            return arch[0] + "." + arch[1]
        return arch[:-1] + "." + arch[-1]

    # 3. GPU compute version
    if hasattr(tvm, "musa"):
        dev = tvm.musa(0)
        if dev.exist:
            return dev.compute_version

    # 4. PyTorch MUSA runtime fallback (useful when TVM MUSA device API
    # is not available in the current build).
    try:
        import torch

        if hasattr(torch, "musa") and torch.musa.is_available():
            props = torch.musa.get_device_properties(0)
            major = getattr(props, "major", None)
            minor = getattr(props, "minor", None)
            if major is not None and minor is not None:
                return f"{int(major)}.{int(minor)}"
    except Exception:
        pass

    raise ValueError(
        "No MUSA architecture was specified or GPU detected."
        "Try specifying it by adding '-arch=mp_xx' to your target."
    )


def parse_musa_compute_version(compute_version) -> tuple[int, int]:
    """Parse compute capability string to divide major and minor version

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "6.0")

    Returns
    -------
    major : int
        major version number
    minor : int
        minor version number
    """
    split_ver = compute_version.split(".")
    try:
        major = int(split_ver[0])
        minor = int(split_ver[1])
        return major, minor
    except (IndexError, ValueError) as err:
        # pylint: disable=raise-missing-from
        raise RuntimeError("Compute version parsing error") from err


def get_musa_arch(compute_version) -> str:
    major, minor = parse_musa_compute_version(compute_version)
    target_arch = str(major * 10 + minor)
    return target_arch


def have_fp16(compute_version):
    """Either fp16 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version: str
        compute capability of a GPU (e.g. "6.0")
    """
    major, _ = parse_musa_compute_version(compute_version)
    return major >= 2


def have_int8(compute_version):
    """Either int8 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "6.1")
    """
    major, _ = parse_musa_compute_version(compute_version)
    return major >= 2


@tvm_ffi.register_global_func("tvm.contrib.mcc.supports_bf16", override=True)
def have_bf16(compute_version):
    """Either bf16 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "8.0")
    """
    major, _ = parse_musa_compute_version(compute_version)
    return major >= 2


@tvm_ffi.register_global_func("tvm.contrib.mcc.supports_fp8", override=True)
def have_fp8(compute_version):
    """Whether fp8 support is provided in the specified compute capability or not

    Parameters
    ----------
    compute_version : str
        GPU capability
    """
    major, _ = parse_musa_compute_version(compute_version)
    # fp8 is supported in S5000 or later architectures.
    return major >= 3


@tvm_ffi.register_global_func("tvm.contrib.mcc.supports_tma", override=True)
def have_tma(target):
    """Whether TMA support is provided in the specified compute capability or not

    Parameters
    ----------
    target : tvm.target.Target
        The compilation target
    """
    if target.kind.name != "musa":
        return False
    compute_version = get_musa_compute_version(target)
    major, _ = parse_musa_compute_version(compute_version)
    # TMA is supported in S5000 or later architectures.
    return major >= 3


def is_qy2(target):
    if target.kind.name != "musa":
        return False
    compute_version = get_musa_compute_version(target)
    major, minor = parse_musa_compute_version(compute_version)
    return major == 2 and minor == 2


def is_ph1(target):
    if target.kind.name != "musa":
        return False
    compute_version = get_musa_compute_version(target)
    major, minor = parse_musa_compute_version(compute_version)
    return major == 3 and minor == 1


def get_mcc_compiler() -> str:
    """Get the path to the mcc compiler"""
    return os.path.join(find_musa_path(), "bin", "mcc")
