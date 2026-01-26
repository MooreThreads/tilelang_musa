from __future__ import annotations
import ctypes
import importlib
import logging
import os
import os.path as osp
import subprocess
import tempfile
from typing import Any

from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.transform import PassConfigKey
from tilelang.contrib.nvcc import get_nvcc_compiler, get_target_arch, get_target_compute_version
from tilelang.contrib.mcc import get_mcc_compiler, get_musa_arch, get_musa_compute_version
from tilelang.contrib.rocm import find_rocm_path, get_rocm_arch
from tilelang.env import TILELANG_TEMPLATE_PATH, env

from .utils import is_cpu_target, is_cuda_target, is_hip_target, is_musa_target

logger = logging.getLogger(__name__)


def _parse_mcc_extra_args() -> list[str]:
    raw = os.environ.get("TILELANG_MCC_EXTRA_ARGS")
    if not raw:
        return []
    args: list[str] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        args.append(chunk)
    return args


try:
    from tilelang.jit.adapter.nvrtc import is_nvrtc_available
    if is_nvrtc_available:
        import cuda.bindings.driver as cuda
        from tilelang.contrib.nvrtc import compile_cuda
except ImportError:
    is_nvrtc_available = False


class LibraryGenerator:
    srcpath: str | None = None
    libpath: str | None = None
    lib_code: str | None = None
    pass_configs: dict[str, Any] | None = None
    compile_flags: list[str] | None = None

    def __init__(self, target: Target, verbose: bool = False):
        self.target = target
        self.verbose = verbose

    def assign_pass_configs(self, pass_configs: dict[str, Any] | None = None):
        self.pass_configs = pass_configs

    def assign_compile_flags(self, compile_flags: list[str] | None = None):
        if compile_flags is None:
            compile_flags = []
        self.compile_flags = compile_flags

    def update_lib_code(self, lib_code: str):
        self.lib_code = lib_code

    # Assume currently we only support CUDA compilation
    def load_lib(self, lib_path: str | None = None):
        if lib_path is None:
            lib_path = self.libpath
        else:
            self.libpath = lib_path
        replace_lib = os.environ.get("TILELANG_REPLACE_LIB")
        if (replace_lib and os.path.isfile(replace_lib)):
            lib_path = replace_lib
        return ctypes.CDLL(lib_path)

    def compile_lib(self, timeout: float = None):
        target = self.target
        verbose = self.verbose

        enable_fast_math = self.pass_configs.get(PassConfigKey.TL_ENABLE_FAST_MATH, False)

        if is_cuda_target(target):
            from tilelang.env import CUTLASS_INCLUDE_DIR
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False)  # noqa: SIM115
            target_arch = get_target_arch(get_target_compute_version(target))
            libpath = src.name.replace(".cu", ".so")

            ptxas_usage_level = self.pass_configs.get(PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL,
                                                      None)
            verbose_ptxas_output = self.pass_configs.get(
                PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT, False)

            command = [
                get_nvcc_compiler(),
                "-std=c++17",
                "-w",  # Disable all warning messages
                "-Xcudafe",
                "--diag_suppress=177",
                "--compiler-options",
                "-fPIC",
                "-lineinfo",
                "--shared",
                src.name,
                "-lcuda",
                "-gencode",
                f"arch=compute_{target_arch},code=sm_{target_arch}",
            ]
            if enable_fast_math:
                command += ["--use_fast_math"]
            if ptxas_usage_level is not None:
                command += [f"--ptxas-options=--register-usage-level={ptxas_usage_level}"]
            if verbose_ptxas_output:
                command += ["--ptxas-options=--verbose"]
            command += [
                "-I" + CUTLASS_INCLUDE_DIR,
            ]
        elif is_musa_target(target):
            from tilelang.env import MUTLASS_INCLUDE_DIR
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".mu", delete=False)  # noqa: SIM115
            target_arch = get_musa_arch(get_musa_compute_version(target))
            libpath = src.name.replace(".mu", ".so")
            opt_level = os.environ.get("TILELANG_OPT_LEVEL")
            command = [
                get_mcc_compiler(),
                src.name,
                "-std=c++17",
                "--shared",
                "-fPIC",
                # "-lmusart",
                # "-lmusa",
                f"--offload-arch=mp_{target_arch}",
            ]
            if opt_level:
                command += ["-Od3"]
            if enable_fast_math:
                command += ["-ffast-math"]
            command += [
                "-mllvm",
                "-mtgpu-alloc-shared-memory-from-zero=1",
            ]
            command += _parse_mcc_extra_args()
            command += [
                "-I" + MUTLASS_INCLUDE_DIR,
            ]

        elif is_hip_target(target):
            from tilelang.env import COMPOSABLE_KERNEL_INCLUDE_DIR
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)  # noqa: SIM115
            libpath = src.name.replace(".cpp", ".so")
            rocm_path = find_rocm_path()
            arch = get_rocm_arch(rocm_path)
            command = [
                "hipcc",
                "-std=c++17",
                "-fPIC",
                f"--offload-arch={arch}",
                "--shared",
                src.name,
            ]
            command += [
                "-I" + COMPOSABLE_KERNEL_INCLUDE_DIR,
            ]
        elif is_cpu_target(target):
            from tilelang.contrib.cc import get_cplus_compiler
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)  # noqa: SIM115
            libpath = src.name.replace(".cpp", ".so")

            command = [get_cplus_compiler(), "-std=c++17", "-fPIC", "-shared", src.name]
            command += [
                "-I" + TILELANG_TEMPLATE_PATH,
            ]
        else:
            raise ValueError(f"Unsupported target: {target}")

        command += [
            "-I" + TILELANG_TEMPLATE_PATH,
        ]

        if self.compile_flags:
            command += [item for flag in self.compile_flags for item in flag.split()]

        command += ["-o", libpath]

        # Check if custom source file is specified via environment variable
        lib_code_to_write = self.lib_code
        custom_source = env.TILELANG_REPLACE_MUSAC
        if custom_source and os.path.isfile(custom_source):
            try:
                with open(custom_source, encoding='utf-8') as f:
                    lib_code_to_write = f.read()
                if verbose:
                    logger.info(f"Replacing Musa C with: {custom_source}")
            except Exception as e:
                logger.warning(
                    f"Failed to read replaced file {custom_source}: {e}, using generated code")
                raise

        src.write(lib_code_to_write)
        src.flush()

        try:
            if verbose:
                print(f"compile_lib compilation command: {' '.join(command)}")
            ret = subprocess.run(command, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Compile kernel failed because of {e}") from e

        if ret.returncode != 0:
            raise RuntimeError(f"Compilation Failed! {command}"
                               f"\n {self.lib_code}")

        self.srcpath = src.name
        self.libpath = libpath

    def remove_lib(self):
        if self.libpath:
            os.remove(self.libpath)
        self.libpath = None

    def get_source_path(self):
        return self.srcpath

    def get_lib_path(self):
        return self.libpath

    def set_lib_path(self, libpath):
        self.libpath = libpath

    def set_src_path(self, srcpath):
        self.srcpath = srcpath


class PyLibraryGenerator(LibraryGenerator):
    host_func: str | None = None
    culib = None
    pymodule = None

    def __init__(self, target: Target, verbose: bool = False):
        if not is_nvrtc_available:
            raise ImportError("cuda-python is not available, nvrtc backend cannot be used. "
                              "Please install cuda-python via `pip install cuda-python` "
                              "if you want to use the nvrtc backend.")
        super().__init__(target, verbose)

    @staticmethod
    def import_from_file(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def update_host_func(self, host_func: str):
        self.host_func = host_func

    def load_lib(self, lib_path: str | None = None):
        if lib_path is None:
            lib_path = self.libpath

        pypath = lib_path.replace(".cubin", ".py")
        self.pymodule = self.import_from_file("kernel", pypath)

        # Ensure the context is valid
        ctx = cuda.cuCtxGetCurrent()[1]
        if cuda.cuCtxGetApiVersion(ctx)[0] != cuda.CUresult.CUDA_SUCCESS:
            import torch
            torch.cuda.synchronize()

        result, self.culib = cuda.cuLibraryLoadFromFile(
            bytes(lib_path, "utf-8"), [], [], 0, [], [], 0)
        assert result == cuda.CUresult.CUDA_SUCCESS, f"Failed to load library: {lib_path}"

    def compile_lib(self, timeout: float = None):
        target = self.target
        verbose = self.verbose
        if is_cuda_target(target):
            from tilelang.env import (CUDA_HOME, CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH)
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False)  # noqa: SIM115
            libpath = src.name.replace(".cu", ".cubin")

            project_root = osp.join(osp.dirname(__file__), "..", "..")
            if CUTLASS_INCLUDE_DIR is None:
                cutlass_path = osp.abspath(osp.join(project_root, "3rdparty/cutlass/include"))
            else:
                cutlass_path = CUTLASS_INCLUDE_DIR

            if TILELANG_TEMPLATE_PATH is None:
                tl_template_path = osp.abspath(osp.join(project_root, "src"))
            else:
                tl_template_path = TILELANG_TEMPLATE_PATH

            cuda_home = CUDA_HOME if CUDA_HOME else "/usr/local/cuda"

            options = [f"-I{tl_template_path}", f"-I{cutlass_path}", f"-I{cuda_home}/include"]
            if self.compile_flags:
                options += [
                    item for flag in self.compile_flags for item in flag.split()
                    if item not in options
                ]

            cubin_bytes = compile_cuda(
                self.lib_code, target_format="cubin", options=options, verbose=verbose)
            with open(libpath, "wb") as f:
                f.write(cubin_bytes)

            src.write(self.lib_code)
            src.flush()

            self.srcpath = src.name
            self.libpath = libpath

            pypath = src.name.replace(".cu", ".py")
            with open(pypath, "w") as f:
                f.write(self.host_func)
        else:
            raise ValueError(f"Unsupported target: {target}")

    def __del__(self):
        if self.culib:
            result = cuda.cuLibraryUnload(self.culib)[0]
            if result != cuda.CUresult.CUDA_SUCCESS:
                logger.warning(f"Failed to unload library: {self.libpath}")
            self.culib = None
