import torch
from tilelang import tvm

IS_CUDA = torch.cuda.is_available()

IS_MPS = False
try:
    IS_MPS = torch.backends.mps.is_available()
except AttributeError:
    print("MPS backend is not available in this PyTorch build.")
except Exception as e:
    print(f"An unexpected error occurred while checking MPS availability: {e}")

IS_MUSA = False
try:
    IS_MUSA = torch.musa.is_available()
except AttributeError:
    print("MUSA backend is not available in this PyTorch build.")
except Exception as e:
    print(f"An unexpected error occurred while checking MUSA availability: {e}")


if IS_CUDA:
    GPUEvent = torch.cuda.Event
elif IS_MUSA:
    GPUEvent = torch.musa.Event
elif IS_MPS:
    GPUEvent = torch.mps.Event


def get_current_device():
    device = None
    if IS_CUDA:
        device = torch.cuda.current_device()
    elif IS_MUSA:
        device = torch.musa.current_device()
    elif IS_MPS:
        device = "mps:0"

    return device


def synchronize():
    if IS_MUSA:
        torch.musa.synchronize()
    else:
        torch.cuda.synchronize()


def get_dl_device(target: str):
    if target == "cuda":
        device = tvm.cuda(0)
    elif target == "musa":
        device = tvm.musa(0)
    elif target == "hip":
        device = tvm.rocm(0)
    else:
        raise ValueError(f"Unsupported device type from {target}")
    return device


def get_pt_device():
    if IS_MUSA:
        device = "musa:0"
    elif IS_MPS:
        device = "mps:0"
    else:
        device = "cuda:0"
    return device
