import math

import torch

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


def get_test_device() -> str:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("Neither MUSA nor CUDA is available")


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },)
def mhc_post_tilelang(hc: int,
                      hidden: int,
                      n_thr: int = 32,
                      h_blk: int = 1024) -> tilelang.JITKernel:
    # rename for shorter code
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def kernel_impl(a: T.Tensor((n, hc, hc), T.float32), b: T.Tensor((n, hc, h), T.bfloat16),
                    c: T.Tensor((n, hc), T.float32), d: T.Tensor((n, h), T.bfloat16), x: T.Tensor(
                        (n, hc, h), T.bfloat16)):
        with T.Kernel(n, threads=n_thr) as i_n:
            x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
            b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
            d_shared = T.alloc_shared(h_blk, T.bfloat16)

            x_local = T.alloc_fragment((hc, h_blk), T.float32)
            b_local = T.alloc_fragment((hc, h_blk), T.float32)
            d_local = T.alloc_fragment(h_blk, T.float32)

            a_local = T.alloc_fragment((hc, hc), T.float32)
            c_local = T.alloc_fragment(hc, T.float32)
            T.copy(a[i_n, 0, 0], a_local)
            T.copy(c[i_n, 0], c_local)

            for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=0):
                T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
                T.copy(d[i_n, i0_h * h_blk], d_shared)

                T.copy(b_shared, b_local)
                T.copy(d_shared, d_local)
                for i_hco, i1_h in T.Parallel(hc, h_blk):
                    x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                    for i_hci in T.serial(hc):
                        x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
                T.copy(x_local, x_shared)

                T.copy(x_shared, x[i_n, 0, i0_h * h_blk])

    return kernel_impl


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(residual)
    kernel = mhc_post_tilelang(residual.shape[-2], residual.shape[-1])
    kernel(comb_res_mix, residual, post_layer_mix.squeeze(-1), x, out)
    return out


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    print(comb_res_mix.shape)
    print(residual.shape)
    term2 = torch.bmm(comb_res_mix.mT, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def generate_test_data(
    n: int,
    h: int,
    hc_mult: int,
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    """Generate test data for post operator."""
    torch.random.manual_seed(42)
    if device is None:
        device = get_test_device()

    x = torch.randn((n, h), dtype=torch.bfloat16, device=device)
    residual = torch.randn((n, hc_mult, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn((n, hc_mult, hc_mult), dtype=torch.float32, device=device)

    return {
        "x": x,
        "residual": residual,
        "post_layer_mix": post_layer_mix,
        "comb_res_mix": comb_res_mix,
    }


def test(n: int, h: int) -> None:
    print(f"Testing mhc_post with {n=} {h=}")
    test_data = generate_test_data(n=n, h=h, hc_mult=4)
    out_tl = mhc_post(**test_data)
    out_ref = mhc_post_ref(**test_data)
    torch.testing.assert_close(out_tl, out_ref)


def benchmark(n: int, h: int) -> None:
    hc_mult = 4
    device = get_test_device()
    test_data = generate_test_data(n=n, h=h, hc_mult=hc_mult, device=device)
    ms = do_bench(lambda: mhc_post(**test_data), warmup=100, rep=100)
    io_bytes = (
        test_data["x"].numel() * 2 + test_data["residual"].numel() * 2 +
        test_data["post_layer_mix"].numel() * 4 + test_data["comb_res_mix"].numel() * 4 +
        n * hc_mult * h * 2)
    # Per output element: one c*d multiply + hc_mult multiply-add accumulations.
    total_flops = n * h * hc_mult * (2 * hc_mult + 1)
    bandwidth_tbps = io_bytes / (ms * 1e-3) / 1e12
    tflops = total_flops / ms * 1e-9
    print(f"[PERF] case=mhc_post device={device} "
          f"params=n={n},h={h},hc_mult={hc_mult}")
    print(f"[PERF] avg_time_ms={ms:.3f} bandwidth_TBps={bandwidth_tbps:.6f} "
          f"tflops={tflops:.6f}")


def main():
    for n in [12288]:
        for h in [2560, 4096, 7168]:
            test(n=n, h=h)
            benchmark(n=n, h=h)


if __name__ == "__main__":
    main()
