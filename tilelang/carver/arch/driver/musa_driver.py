from __future__ import annotations

try:
    import torch.musa._CudaDeviceProperties as _MusaDeviceProperties
except ImportError:
    _MusaDeviceProperties = type("DummyMusaDeviceProperties", (), {})


class musaDeviceAttrNames:
    r"""
    refer to MUSA runtime attribute ids.
    Keep ids aligned with CUDA-compatible values used by tests.
    """

    musaDevAttrMaxThreadsPerBlock: int = 1
    musaDevAttrMaxSharedMemoryPerBlock: int = 8
    musaDevAttrMaxRegistersPerBlock: int = 12
    musaDevAttrMultiProcessorCount: int = 16
    musaDevAttrMaxSharedMemoryPerMultiprocessor: int = 81
    musaDevAttrMaxPersistingL2CacheSize: int = 108


def _get_attr_int(obj, *names) -> int | None:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                continue
    return None


def get_musa_device_properties(device_id: int = 0) -> _MusaDeviceProperties | None:
    try:
        import torch.musa

        if not torch.musa.is_available():
            return None
        return torch.musa.get_device_properties(device_id)
    except Exception:
        return None


def get_device_name(device_id: int = 0) -> str | None:
    prop = get_musa_device_properties(device_id)
    if prop is not None:
        return getattr(prop, "name", None)
    return None


def get_device_attribute(attr: int, device_id: int = 0) -> int | None:
    prop = get_musa_device_properties(device_id)
    if prop is None:
        return None

    attr_map = {
        musaDeviceAttrNames.musaDevAttrMaxThreadsPerBlock: _get_attr_int(prop, "max_threads_per_block"),
        musaDeviceAttrNames.musaDevAttrMaxSharedMemoryPerBlock: _get_attr_int(
            prop, "shared_memory_per_block", "shared_mem_per_block"
        ),
        musaDeviceAttrNames.musaDevAttrMaxRegistersPerBlock: _get_attr_int(
            prop, "max_registers_per_block", "regs_per_block"
        ),
        musaDeviceAttrNames.musaDevAttrMultiProcessorCount: _get_attr_int(prop, "multi_processor_count"),
        musaDeviceAttrNames.musaDevAttrMaxSharedMemoryPerMultiprocessor: _get_attr_int(
            prop,
            "shared_memory_per_multiprocessor",
            "shared_mem_per_multiprocessor",
        ),
        musaDeviceAttrNames.musaDevAttrMaxPersistingL2CacheSize: _get_attr_int(
            prop,
            "max_persisting_l2_cache_size",
            "persisting_l2_cache_max_size",
        ),
    }
    return attr_map.get(attr)


def get_shared_memory_per_block(device_id: int = 0, format: str = "bytes") -> int | None:
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    shared_mem = get_device_attribute(musaDeviceAttrNames.musaDevAttrMaxSharedMemoryPerBlock, device_id)
    if shared_mem is None:
        return None
    if format == "bytes":
        return shared_mem
    if format == "kb":
        return shared_mem // 1024
    if format == "mb":
        return shared_mem // (1024 * 1024)
    raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb")


def get_max_dynamic_shared_size_bytes(device_id: int = 0, format: str = "bytes") -> int | None:
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    shared_mem = get_device_attribute(musaDeviceAttrNames.musaDevAttrMaxSharedMemoryPerMultiprocessor, device_id)
    if shared_mem is None:
        return None
    if format == "bytes":
        return shared_mem
    if format == "kb":
        return shared_mem // 1024
    if format == "mb":
        return shared_mem // (1024 * 1024)
    raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb")


def get_persisting_l2_cache_max_size(device_id: int = 0) -> int | None:
    return get_device_attribute(musaDeviceAttrNames.musaDevAttrMaxPersistingL2CacheSize, device_id)


def get_num_sms(device_id: int = 0) -> int | None:
    return get_device_attribute(musaDeviceAttrNames.musaDevAttrMultiProcessorCount, device_id)


def get_registers_per_block(device_id: int = 0) -> int | None:
    return get_device_attribute(musaDeviceAttrNames.musaDevAttrMaxRegistersPerBlock, device_id)
