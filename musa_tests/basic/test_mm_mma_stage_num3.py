import itertools
import pytest
import mm_mma_stage_num3

MNK_CASES = [
    (256, 256, 512),
    # (512, 512, 512),
    # (512, 256, 512),
    # (256, 512, 512),
]

BLOCK_CASES = [
    # m32
    (32, 128, 64),
    (32, 128, 32),
    (32, 64, 64),
    (32, 64, 32),
    (32, 32, 64),
    (32, 32, 32),
    (32, 32, 128),
    (32, 32, 256),
    # m64
    (64, 128, 64),
    (64, 128, 32),
    (64, 64, 64),
    (64, 64, 32),
    (64, 32, 64),
    (64, 32, 32),
    (64, 64, 128),
    (64, 64, 256),
    # m128
    (128, 128, 64),
    (128, 128, 32),
    (128, 64, 64),
    (128, 64, 32),
    (128, 32, 64),
    (128, 32, 32),
    # m16n64/m64n16
    (64, 16, 64),
    (64, 16, 32),
    (16, 64, 64),
    (16, 64, 32),
]

TYPE_CASES = [
    ("float16", "float32"),
    ("bfloat16", "float32"),
    ("float8_e4m3", "float32"),
    # ("tfloat32", "float32"),
]

WARP_CASES = [(4, "m")]

TEST_CASES = [
    pytest.param(
        M,
        N,
        K,
        bm,
        bn,
        bk,
        dtype,
        acc_type,
        warp,
        policy,
        id=f"M{M}-N{N}-K{K}-bm{bm}-bn{bn}-bk{bk}-{dtype}-{acc_type}-warp{warp}-{policy}",
    ) for (M, N, K), (bm, bn, bk), (dtype, acc_type), (
        warp, policy) in itertools.product(MNK_CASES, BLOCK_CASES, TYPE_CASES, WARP_CASES)
]

SPECIAL_CASES = [
    (1024, 1024, 1024, 32, 128, 16, "float16", "float32", 4, "m"),
    (1024, 1024, 1024, 32, 64, 16, "float16", "float32", 4, "m"),
    (1024, 1024, 1024, 32, 32, 16, "float16", "float32", 4, "m"),
    (1024, 1024, 1024, 64, 128, 16, "float16", "float32", 4, "m"),
    (1024, 1024, 1024, 64, 64, 16, "float16", "float32", 4, "m"),
    (1024, 1024, 1024, 64, 32, 16, "float16", "float32", 4, "m"),
    (1024, 1024, 1024, 128, 128, 16, "float16", "float32", 4, "m"),
    (1024, 1024, 1024, 128, 64, 16, "float16", "float32", 4, "m"),
    (1024, 1024, 1024, 128, 32, 16, "float16", "float32", 4, "m"),
    (1024, 1024, 1024, 64, 16, 16, "float16", "float32", 4, "m"),
    (1024, 1024, 1024, 16, 64, 16, "float16", "float32", 4, "m"),
    (1024, 1024, 1024, 128, 128, 128, "float16", "float32", 4, "m"),
    (512, 512, 512, 256, 128, 64, "float16", "float32", 8, "m"),
    (512, 512, 512, 256, 128, 64, "float16", "float32", 8, "n"),
    (512, 512, 512, 128, 256, 64, "float16", "float32", 8, "m"),
    (512, 512, 512, 128, 256, 64, "float16", "float32", 8, "n"),
    (512, 512, 512, 256, 256, 64, "float16", "float32", 16, "square"),
    (512, 512, 512, 256, 256, 64, "float16", "float32", 16, "m"),
    (512, 512, 512, 256, 256, 64, "float16", "float32", 16, "n"),
]

TEST_CASES += [
    pytest.param(
        M,
        N,
        K,
        bm,
        bn,
        bk,
        dtype,
        acc_type,
        warp,
        policy,
        id=f"M{M}-N{N}-K{K}-bm{bm}-bn{bn}-bk{bk}-{dtype}-{acc_type}-warp{warp}-{policy}",
    ) for (M, N, K, bm, bn, bk, dtype, acc_type, warp, policy) in SPECIAL_CASES
]


@pytest.mark.parametrize(
    "M,N,K,bm,bn,bk,dtype,acc_type,warp,policy",
    TEST_CASES,
)
def test_mm_mma(M, N, K, bm, bn, bk, dtype, acc_type, warp, policy):
    mm_mma_stage_num3.run(M, N, K, bm, bn, bk, dtype, acc_type, warp, policy, verbose=False)
