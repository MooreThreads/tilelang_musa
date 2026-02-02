import itertools
import pytest
import mm_mma_stage_num1

MNK_CASES = [
    (256, 256, 128),
]

BLOCK_CASES = [
    # m32
    (32, 128, 64),
    (32, 128, 32),
    (32, 128, 16),
    (32, 64, 64),
    (32, 64, 32),
    (32, 64, 16),
    (32, 32, 64),
    (32, 32, 32),
    (32, 32, 16),
    (32, 32, 128), # right

    # m64
    (64, 128, 64),
    (64, 128, 32),
    (64, 128, 16),
    (64, 64, 64),
    (64, 64, 32),
    (64, 64, 16),
    (64, 32, 64),
    (64, 32, 32),
    (64, 32, 16),

    (64, 64, 128), # right
    (64, 64, 256),

    # m128
    (128, 128, 64),
    (128, 128, 32),
    (128, 128, 16),
    (128, 64, 64),
    (128, 64, 32),
    (128, 64, 16),
    (128, 32, 64),
    (128, 32, 32),
    (128, 32, 16),

    # m16n64/m64n16
    (64, 16, 64),
    (64, 16, 32),
    (64, 16, 16),
    (16, 64, 64),
    (16, 64, 32),
    (16, 64, 16),
]

TYPE_CASES = [
    ("float16", "float32"),
    ("bfloat16", "float32"),
    # ("tfloat32", "float32"),
]

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
        id=f"M{M}-N{N}-K{K}-bm{bm}-bn{bn}-bk{bk}-{dtype}-{acc_type}",
    )
    for (M, N, K), (bm, bn, bk), (dtype, acc_type) in itertools.product(
        MNK_CASES, BLOCK_CASES, TYPE_CASES
    )
]

SPECIAL_CASES = [
    (1024, 1024, 1024, 32, 32, 32, "float16", "float32"),
    (1024, 1024, 1024, 32, 32, 32, "bfloat16", "float32"),
    (1024, 1024, 1024, 128, 128, 256, "float16", "float32"),
]

TEST_CASES += [
    pytest.param(M, N, K, bm, bn, bk, dtype, acc_type,
                 id=f"M{M}-N{N}-K{K}-bm{bm}-bn{bn}-bk{bk}-{dtype}-{acc_type}")
    for (M, N, K, bm, bn, bk, dtype, acc_type) in SPECIAL_CASES
]



@pytest.mark.parametrize(
    "M,N,K,bm,bn,bk,dtype,acc_type",
    TEST_CASES,
)
def test_mm_mma(M, N, K, bm, bn, bk, dtype, acc_type):
    mm_mma_stage_num1.run(M, N, K, bm, bn, bk, dtype, acc_type, verbose=False)
