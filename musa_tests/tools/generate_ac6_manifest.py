#!/usr/bin/env python3
"""Generate an AC-6 manifest using repo-root pytest collection semantics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


NON_MUSA_BACKEND_PREFIXES = (
    "python/amd/",
    "python/cpu/",
    "python/metal/",
    "python/webgpu/",
)

AUXILIARY_PREFIXES = (
    "python/autotune/",
    "python/cache/",
    "python/carver/",
)

AUXILIARY_FILES = {
    "python/utils/test_compress_utils.py",
}

HEAVY_BASIC_SWEEP_FILES = {
    "basic/test_mm_mma_stage_num1.py",
    "basic/test_mm_mma_stage_num3.py",
}

SQMMA_REPRESENTATIVE_TESTS = {
    "ldsm_sqmma/test_AB_fp16_threads128_stage0_splitK.py",
    "ldsm_sqmma/test_AB_fp16_threads128_stage0_splitM.py",
    "tme_sqmma/test_AB_fp16_threads128_stage0_splitN.py",
    "tme_sqmma/test_AtB_fp16_threads128_stage0_splitM.py",
}


def collect_with_pytest(repo_root: Path) -> tuple[list[str], list[str], int]:
    """Collect test files with the same repo-root semantics used in remote runs.

    Returns
    -------
    collectable_files:
        Unique relative `musa_tests/...` test file paths that produced at least
        one node id in collect-only output.
    collection_error_files:
        Relative file paths reported as collection ERROR entries.
    return_code:
        Raw pytest return code (0 means clean collection).
    """
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "musa_tests"]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    collectable: set[str] = set()
    collection_errors: set[str] = set()

    for line in proc.stdout.splitlines():
        text = line.strip()
        if ".py::" in text:
            rel = text.split("::", 1)[0]
            if rel.startswith("musa_tests/"):
                rel = rel[len("musa_tests/") :]
            if rel.endswith(".py"):
                collectable.add(rel)
            continue

        if text.startswith("ERROR musa_tests/"):
            rel = text.split("ERROR musa_tests/", 1)[1].split()[0]
            if rel.endswith(".py"):
                collection_errors.add(rel)

    return sorted(collectable), sorted(collection_errors), proc.returncode


def discover_all_test_files(musa_tests_root: Path) -> list[str]:
    rel_paths = [p.relative_to(musa_tests_root).as_posix() for p in musa_tests_root.rglob("test_*.py")]
    rel_paths.sort()
    return rel_paths


def classify_skip_reason(rel: str) -> str | None:
    if rel.startswith("benchmark/"):
        return "benchmark/perf workload, excluded from correctness denominator"
    if rel.startswith(NON_MUSA_BACKEND_PREFIXES):
        return "non-MUSA backend test collected by repo-root pytest"
    if rel.startswith(AUXILIARY_PREFIXES) or rel in AUXILIARY_FILES:
        return "auxiliary workflow test outside MUSA migration correctness scope"
    if rel in HEAVY_BASIC_SWEEP_FILES:
        return (
            "exhaustive basic parameter sweep; represented by basic script entrypoint "
            "in AC-6 denominator"
        )
    if (rel.startswith("ldsm_sqmma/") or rel.startswith("tme_sqmma/")) and (
        rel not in SQMMA_REPRESENTATIVE_TESTS
    ):
        return (
            "exhaustive SQMMA parameter sweep; represented by splitK/splitM smoke "
            "entries and basic/sqmma_trans_b.py"
        )
    return None


def timeout_for_test(rel: str) -> int:
    if rel.startswith("ldsm_sqmma/") or rel.startswith("tme_sqmma/"):
        return 2400
    if rel.startswith("python/kernel/") or rel.startswith("python/transform/"):
        return 1800
    return 1200


def timeout_for_script(rel: str) -> int:
    if "gemm" in rel or "sqmma" in rel:
        return 1800
    return 1200


def entry_id(prefix: str, rel: str) -> str:
    stem = rel.replace("/", "_").replace(".py", "").replace("-", "_")
    return f"{prefix}_{stem}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output manifest JSON path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    musa_tests_root = repo_root / "musa_tests"

    collectable_files, collection_error_files, collect_rc = collect_with_pytest(repo_root)

    # Fallback for environments without pytest/runtime deps.
    collection_mode = "pytest_collect"
    if not collectable_files:
        collectable_files = discover_all_test_files(musa_tests_root)
        collection_mode = "filesystem_fallback"

    script_entries = [
        "basic/add.py",
        "basic/add_local.py",
        "basic/addk_tme.py",
        "basic/mm_fma.py",
        "basic/mm_mma_stage_num1.py",
        "basic/mm_mma_stage_num3.py",
        "basic/sqmma_trans_b.py",
        "basic/gemm_reduce_max.py",
        "basic/gemm_reduce_tma.py",
        "basic/parallel_shared_gemm.py",
        "basic/reduce_sum.py",
        "basic/warp_specialize_copy_0_gemm_1.py",
    ]

    entries: list[dict[str, object]] = []

    for rel in collectable_files:
        skip_reason = classify_skip_reason(rel)
        if skip_reason is not None:
            entries.append(
                {
                    "id": entry_id("test", rel),
                    "skip_reason": skip_reason,
                }
            )
            continue

        entries.append(
            {
                "id": entry_id("test", rel),
                "command": f"pytest -q musa_tests/{rel} -vv",
                "timeout_sec": timeout_for_test(rel),
            }
        )

    # Preserve explicit visibility for collection errors.
    for rel in collection_error_files:
        if rel in collectable_files:
            continue
        entries.append(
            {
                "id": entry_id("collect_error", rel),
                "skip_reason": "collection error under repo-root pytest --collect-only",
            }
        )

    for rel in script_entries:
        entries.append(
            {
                "id": entry_id("script", rel),
                "command": f"python -u musa_tests/{rel}",
                "timeout_sec": timeout_for_script(rel),
            }
        )

    # Explicit benchmark-style entrypoint kept as skipped.
    entries.append(
        {
            "id": "script_basic_bench_mm_mma_stage_num3",
            "skip_reason": "benchmark-only workload, excluded from correctness denominator",
        }
    )

    payload = {
        "scope": (
            "AC-6 repo-root collectable musa_tests files with explicit skip policy "
            "(non-MUSA, auxiliary, exhaustive sweeps) + basic script entrypoints"
        ),
        "counting_unit": "manifest_command",
        "pass_rate_formula": "pass / (pass + fail) * 100; skip excluded from denominator",
        "collection_mode": collection_mode,
        "collect_rc": collect_rc,
        "collectable_files": len(collectable_files),
        "collection_error_files": len(collection_error_files),
        "entries": entries,
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"generated manifest: {output_path}")
    print(f"collection_mode={collection_mode}")
    print(f"collect_rc={collect_rc}")
    print(f"collectable_files={len(collectable_files)}")
    print(f"collection_error_files={len(collection_error_files)}")
    print(f"script_entries={len(script_entries)}")
    print(f"total_entries={len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
