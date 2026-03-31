#!/usr/bin/env python3
"""Run a manifest of test commands and produce unified pass/fail/skip statistics."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EntryResult:
    id: str
    status: str
    command: Optional[str]
    skip_reason: Optional[str]
    returncode: Optional[int]
    duration_sec: float
    log_path: Optional[str]
    attempts: int


def _load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "entries" not in data:
        raise ValueError("manifest must be a JSON object with an 'entries' list")
    if not isinstance(data["entries"], list):
        raise ValueError("manifest['entries'] must be a list")
    return data


def _is_pytest_command(command: str) -> bool:
    pytest_patterns = [
        r"(^|\s)pytest(\s|$)",
        r"(^|\s)python(\d+)?\s+-m\s+pytest(\s|$)",
    ]
    return any(re.search(pattern, command) for pattern in pytest_patterns)


def _detect_pytest_status_from_log(log_path: Path) -> str:
    if not log_path.exists():
        return "pass"
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    # Treat XPASS/XFAIL as non-pass accounting outcomes so AC-6 does not
    # overstate real green coverage.
    if re.search(r"\bXPASS\b", text) or "xpassed" in text.lower():
        return "xpass"
    if re.search(r"\bXFAIL\b", text) or "xfailed" in text.lower():
        return "xfail"
    lower = text.lower()
    has_skip = re.search(r"\b\d+\s+skipped\b", lower)
    has_pass = re.search(r"\b\d+\s+passed\b", lower)
    has_fail = re.search(r"\b\d+\s+failed\b", lower)
    if has_skip and not has_pass and not has_fail:
        return "skip"
    return "pass"


def _run_entry(entry: Dict[str, Any], log_dir: Path, workdir: Path, index: int) -> EntryResult:
    entry_id = str(entry.get("id", f"entry_{index}"))
    command = entry.get("command")
    skip_reason = entry.get("skip_reason")
    timeout = entry.get("timeout_sec")
    retries = int(entry.get("retries", 0))

    if skip_reason:
        return EntryResult(
            id=entry_id,
            status="skip",
            command=command,
            skip_reason=str(skip_reason),
            returncode=None,
            duration_sec=0.0,
            log_path=None,
            attempts=0,
        )

    if not isinstance(command, str) or not command.strip():
        return EntryResult(
            id=entry_id,
            status="fail",
            command=None,
            skip_reason="missing command and no skip_reason",
            returncode=None,
            duration_sec=0.0,
            log_path=None,
            attempts=0,
        )

    started = time.time()
    last_returncode: Optional[int] = None
    last_log_path: Optional[Path] = None
    attempts = retries + 1

    for attempt in range(1, attempts + 1):
        suffix = f".attempt{attempt}" if attempts > 1 else ""
        log_path = log_dir / f"{index:02d}_{entry_id}{suffix}.log"
        last_log_path = log_path
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"$ {command}\n\n")
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(workdir),
                    shell=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=int(timeout) if timeout is not None else None,
                    check=False,
                    executable="/bin/bash",
                )
                last_returncode = completed.returncode
                if last_returncode == 0:
                    status = "pass"
                    if _is_pytest_command(command):
                        status = _detect_pytest_status_from_log(log_path)
                    return EntryResult(
                        id=entry_id,
                        status=status,
                        command=command,
                        skip_reason=None,
                        returncode=0,
                        duration_sec=time.time() - started,
                        log_path=str(log_path),
                        attempts=attempt,
                    )
            except subprocess.TimeoutExpired:
                log.write(f"\n[TIMEOUT] exceeded timeout_sec={timeout}\n")
                last_returncode = 124

    return EntryResult(
        id=entry_id,
        status="fail",
        command=command,
        skip_reason=None,
        returncode=last_returncode,
        duration_sec=time.time() - started,
        log_path=str(last_log_path) if last_log_path else None,
        attempts=attempts,
    )


def _calc_summary(results: List[EntryResult]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.status == "pass")
    xfailed = sum(1 for r in results if r.status == "xfail")
    xpassed = sum(1 for r in results if r.status == "xpass")
    failed = sum(1 for r in results if r.status == "fail") + xfailed + xpassed
    skipped = sum(1 for r in results if r.status == "skip")
    denom = passed + failed
    pass_rate = 100.0 if denom == 0 else (passed / denom) * 100.0
    return {
        "total": total,
        "pass": passed,
        "fail": failed,
        "skip": skipped,
        "xfail": xfailed,
        "xpass": xpassed,
        "pass_rate": round(pass_rate, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to JSON manifest")
    parser.add_argument("--workdir", default=".", help="Working directory to run commands")
    parser.add_argument("--log-dir", required=True, help="Directory for per-entry logs")
    parser.add_argument("--output-json", required=True, help="Summary JSON output path")
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Continue running all entries even if some fail",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    workdir = Path(args.workdir).resolve()
    log_dir = Path(args.log_dir).resolve()
    output_json = Path(args.output_json).resolve()

    log_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(manifest_path)
    entries: List[Dict[str, Any]] = manifest["entries"]
    results: List[EntryResult] = []

    started = time.time()
    for idx, entry in enumerate(entries, start=1):
        result = _run_entry(entry, log_dir=log_dir, workdir=workdir, index=idx)
        results.append(result)
        print(
            f"[{idx:02d}/{len(entries):02d}] {result.id}: {result.status}"
            + (f" (rc={result.returncode})" if result.returncode is not None else "")
            + (f" (attempts={result.attempts})" if result.attempts > 1 else "")
        )
        if result.status == "fail" and not args.continue_on_fail:
            break

    summary = _calc_summary(results)
    payload = {
        "scope": manifest.get("scope", "unspecified"),
        "counting_unit": manifest.get("counting_unit", "manifest_command"),
        "pass_rate_formula": manifest.get(
            "pass_rate_formula",
            "pass / (pass + fail) * 100; skip excluded from denominator",
        ),
        "started_at_epoch": started,
        "duration_sec": round(time.time() - started, 2),
        "summary": summary,
        "results": [
            {
                "id": r.id,
                "status": r.status,
                "command": r.command,
                "skip_reason": r.skip_reason,
                "returncode": r.returncode,
                "duration_sec": round(r.duration_sec, 2),
                "log_path": r.log_path,
                "attempts": r.attempts,
            }
            for r in results
        ],
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(
        "summary: "
        f"total={summary['total']} pass={summary['pass']} fail={summary['fail']} "
        f"skip={summary['skip']} pass_rate={summary['pass_rate']}%"
    )
    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
