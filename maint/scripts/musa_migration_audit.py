#!/usr/bin/env python3
"""Audit commit-level migration coverage from tilelang_musa_6 to tilelang.

This tool provides commit-level traceability for a feature-bucket migration:
1. Enumerate source commits in a configured source range.
2. Compute stable patch-id for source and target commits.
3. Auto-match source commits by patch-id when possible.
4. Apply manual overrides from a CSV ledger.
5. Optionally sync a full per-commit manual ledger (one source commit per row).
6. Emit a full ledger CSV and a markdown summary report.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Sequence


ALLOWED_MANUAL_STATUSES = {
    "pending",
    "auto_patchid_match",
    "ported",
    "adapted",
    "dropped",
    "deferred",
}
UNRESOLVED_STATUSES = {"pending", "unmapped"}
MANUAL_LEDGER_FIELDS = [
    "src_commit",
    "src_subject",
    "status",
    "target_commit",
    "evidence",
    "note",
    "last_updated",
]


@dataclass(frozen=True)
class CommitInfo:
    commit: str
    subject: str


def run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}"
        )
    return proc.stdout


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_commits(repo: Path, rev_range: str) -> List[CommitInfo]:
    output = run_cmd(
        ["git", "-C", str(repo), "log", "--reverse", "--format=%H%x09%s", rev_range]
    ).strip()
    if not output:
        return []
    commits: List[CommitInfo] = []
    for line in output.splitlines():
        parts = line.split("\t", 1)
        commit = parts[0].strip()
        subject = parts[1].strip() if len(parts) > 1 else ""
        commits.append(CommitInfo(commit=commit, subject=subject))
    return commits


def get_patch_id(repo: Path, commit: str) -> Optional[str]:
    show_proc = subprocess.Popen(
        ["git", "-C", str(repo), "show", "--pretty=format:", "--no-color", commit],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    assert show_proc.stdout is not None
    patch_proc = subprocess.run(
        ["git", "patch-id", "--stable"],
        stdin=show_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    show_proc.stdout.close()
    show_proc.wait()
    if patch_proc.returncode != 0:
        return None
    line = patch_proc.stdout.strip().splitlines()
    if not line:
        return None
    return line[0].split()[0]


def git_has_commit(repo: Path, commit: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", f"{commit}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
    )
    return proc.returncode == 0


def load_manual_ledger(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"src_commit", "status", "target_commit", "evidence", "note"}
        if reader.fieldnames is None:
            return {}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise RuntimeError(
                f"manual ledger missing required columns: {sorted(missing)}"
            )
        entries: Dict[str, Dict[str, str]] = {}
        for row in reader:
            src_commit = row.get("src_commit", "").strip()
            if not src_commit:
                continue
            if src_commit in entries:
                raise RuntimeError(f"duplicate src_commit in manual ledger: {src_commit}")
            entries[src_commit] = {
                "src_subject": row.get("src_subject", "").strip(),
                "status": row.get("status", "").strip(),
                "target_commit": row.get("target_commit", "").strip(),
                "evidence": row.get("evidence", "").strip(),
                "note": row.get("note", "").strip(),
                "last_updated": row.get("last_updated", "").strip(),
            }
        return entries


def write_manual_ledger(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANUAL_LEDGER_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sync_manual_ledger(
    path: Path,
    source_commits: List[CommitInfo],
    manual_entries: Dict[str, Dict[str, str]],
    default_status: str,
) -> Dict[str, Dict[str, str]]:
    source_set = {c.commit for c in source_commits}
    out_rows: List[Dict[str, str]] = []
    for commit in source_commits:
        existing = manual_entries.get(commit.commit, {})
        status = existing.get("status", "").strip() or default_status
        last_updated = existing.get("last_updated", "").strip()
        if not last_updated:
            last_updated = utc_now_str()
        out_rows.append(
            {
                "src_commit": commit.commit,
                "src_subject": existing.get("src_subject", "").strip() or commit.subject,
                "status": status,
                "target_commit": existing.get("target_commit", "").strip(),
                "evidence": existing.get("evidence", "").strip(),
                "note": existing.get("note", "").strip(),
                "last_updated": last_updated,
            }
        )

    write_manual_ledger(path, out_rows)

    dropped_entries = sorted(set(manual_entries).difference(source_set))
    if dropped_entries:
        print(
            "warning: manual ledger rows outside source range were dropped: "
            f"{len(dropped_entries)}",
            file=sys.stderr,
        )

    refreshed: Dict[str, Dict[str, str]] = {}
    for row in out_rows:
        refreshed[row["src_commit"]] = {
            "src_subject": row["src_subject"],
            "status": row["status"],
            "target_commit": row["target_commit"],
            "evidence": row["evidence"],
            "note": row["note"],
            "last_updated": row["last_updated"],
        }
    return refreshed


def validate_manual_entries(
    source_repo: Path,
    target_repo: Path,
    manual_entries: Dict[str, Dict[str, str]],
) -> None:
    for src_commit, entry in manual_entries.items():
        if not git_has_commit(source_repo, src_commit):
            raise RuntimeError(f"manual ledger src_commit not found: {src_commit}")

        status = entry.get("status", "")
        if not status:
            raise RuntimeError(f"manual ledger status is empty for src_commit: {src_commit}")
        if status not in ALLOWED_MANUAL_STATUSES:
            raise RuntimeError(
                f"manual ledger status `{status}` is invalid for src_commit: {src_commit}"
            )

        target_commit = entry.get("target_commit", "")
        if target_commit and not git_has_commit(target_repo, target_commit):
            raise RuntimeError(
                "manual ledger target_commit not found in target repo: "
                f"{target_commit} (src={src_commit})"
            )

        if status in {"ported", "adapted"} and not target_commit:
            raise RuntimeError(
                f"manual ledger status `{status}` requires target_commit (src={src_commit})"
            )

        if status in {"dropped", "deferred"} and not entry.get("note", ""):
            raise RuntimeError(
                f"manual ledger status `{status}` requires note (src={src_commit})"
            )


def write_ledger(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "src_commit",
        "src_subject",
        "src_patch_id",
        "status",
        "target_commit",
        "auto_patchid_match",
        "evidence",
        "note",
        "decision_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(
    path: Path,
    source_repo: Path,
    source_range: str,
    target_repo: Path,
    target_range: str,
    rows: List[Dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    by_status: Dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1

    auto_matched = [r for r in rows if r["decision_source"] == "auto"]
    manual = [r for r in rows if r["decision_source"] == "manual"]
    manual_decided = [r for r in manual if r["status"] != "pending"]
    pending = [r for r in rows if r["status"] == "pending"]
    unmapped = [r for r in rows if r["status"] == "unmapped"]
    unresolved = [r for r in rows if r["status"] in UNRESOLVED_STATUSES]
    explained = total - len(unresolved)
    explained_ratio = (100.0 * explained / total) if total else 100.0

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    lines: List[str] = []
    lines.append("# MUSA Commit Migration Audit Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{generated_at}`")
    lines.append(f"- Source repo: `{source_repo}`")
    lines.append(f"- Source range: `{source_range}`")
    lines.append(f"- Target repo: `{target_repo}`")
    lines.append(f"- Target range: `{target_range}`")
    lines.append(f"- Total source commits: `{total}`")
    lines.append(f"- Auto patch-id matches: `{len(auto_matched)}`")
    lines.append(f"- Manual ledger rows: `{len(manual)}`")
    lines.append(f"- Manual decisions (non-pending): `{len(manual_decided)}`")
    lines.append(f"- Pending: `{len(pending)}`")
    lines.append(f"- Unmapped: `{len(unmapped)}`")
    lines.append(f"- Explained coverage: `{explained}/{total}` ({explained_ratio:.2f}%)")
    lines.append("")
    lines.append("## Status Summary")
    lines.append("")
    lines.append("| status | count |")
    lines.append("| --- | ---: |")
    for status in sorted(by_status):
        lines.append(f"| `{status}` | {by_status[status]} |")
    lines.append("")
    lines.append("## Unresolved Commits (pending + unmapped, first 50)")
    lines.append("")
    lines.append("| status | src_commit | subject |")
    lines.append("| --- | --- | --- |")
    for row in unresolved[:50]:
        lines.append(
            f"| `{row['status']}` | `{row['src_commit']}` | {row['src_subject']} |"
        )
    if not unresolved:
        lines.append("| _none_ | _none_ | _none_ |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `auto` means patch-id exact match against target commit diff; this may miss adapted ports."
    )
    lines.append(
        "- `manual` means commit status/evidence is provided in `migration_commit_manual.csv`."
    )
    lines.append(
        "- `pending` means known source commit but not yet decided (ported/adapted/dropped/deferred)."
    )
    lines.append(
        "- Final sign-off should require both `pending=0` and `unmapped=0`."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=Path("../tilelang_musa_6"),
        help="Path to source repo",
    )
    parser.add_argument(
        "--source-range",
        default="47039f0..HEAD",
        help="Source commit range",
    )
    parser.add_argument(
        "--target-repo",
        type=Path,
        default=Path("."),
        help="Path to target repo",
    )
    parser.add_argument(
        "--target-range",
        default="0765d1d6..HEAD",
        help="Target commit range",
    )
    parser.add_argument(
        "--manual-ledger",
        type=Path,
        default=Path("docs/musa/migration_commit_manual.csv"),
        help="Manual decision ledger CSV",
    )
    parser.add_argument(
        "--sync-manual-ledger",
        action="store_true",
        help="Rewrite manual ledger to include all source commits in source order",
    )
    parser.add_argument(
        "--manual-default-status",
        default="pending",
        help="Default status for new source commits when syncing manual ledger",
    )
    parser.add_argument(
        "--output-ledger",
        type=Path,
        default=Path("docs/musa/migration_commit_ledger.csv"),
        help="Output full ledger CSV",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("docs/musa/migration_commit_report.md"),
        help="Output markdown report",
    )
    parser.add_argument(
        "--require-no-unmapped",
        action="store_true",
        help="Exit non-zero when any source commit remains unmapped",
    )
    parser.add_argument(
        "--require-no-pending",
        action="store_true",
        help="Exit non-zero when any source commit remains pending",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    source_repo = args.source_repo.resolve()
    target_repo = args.target_repo.resolve()
    manual_ledger = args.manual_ledger.resolve()
    output_ledger = args.output_ledger.resolve()
    output_report = args.output_report.resolve()

    if args.manual_default_status not in ALLOWED_MANUAL_STATUSES:
        raise RuntimeError(
            f"manual default status must be one of {sorted(ALLOWED_MANUAL_STATUSES)}"
        )

    source_commits = get_commits(source_repo, args.source_range)
    target_commits = get_commits(target_repo, args.target_range)

    target_patch_map: Dict[str, List[str]] = {}
    for commit in target_commits:
        patch_id = get_patch_id(target_repo, commit.commit)
        if not patch_id:
            continue
        target_patch_map.setdefault(patch_id, []).append(commit.commit)

    manual_entries = load_manual_ledger(manual_ledger)
    if args.sync_manual_ledger:
        manual_entries = sync_manual_ledger(
            manual_ledger,
            source_commits,
            manual_entries,
            args.manual_default_status,
        )
        print(f"manual ledger synced: {manual_ledger}")

    validate_manual_entries(source_repo, target_repo, manual_entries)

    rows: List[Dict[str, str]] = []
    for commit in source_commits:
        src_patch_id = get_patch_id(source_repo, commit.commit) or ""
        auto_matches = target_patch_map.get(src_patch_id, []) if src_patch_id else []
        auto_match_str = ",".join(auto_matches)
        manual = manual_entries.get(commit.commit)
        if manual:
            status = manual["status"]
            target_commit = manual["target_commit"]
            evidence = manual["evidence"]
            note = manual["note"]
            decision_source = "manual"
            if status == "auto_patchid_match" and not target_commit and auto_matches:
                target_commit = auto_matches[0]
                if not evidence:
                    evidence = "patch-id"
        elif auto_matches:
            status = "auto_patchid_match"
            target_commit = auto_matches[0]
            evidence = "patch-id"
            note = ""
            decision_source = "auto"
        else:
            status = "unmapped"
            target_commit = ""
            evidence = ""
            note = ""
            decision_source = "none"

        rows.append(
            {
                "src_commit": commit.commit,
                "src_subject": commit.subject,
                "src_patch_id": src_patch_id,
                "status": status,
                "target_commit": target_commit,
                "auto_patchid_match": auto_match_str,
                "evidence": evidence,
                "note": note,
                "decision_source": decision_source,
            }
        )

    write_ledger(output_ledger, rows)
    write_report(
        output_report,
        source_repo,
        args.source_range,
        target_repo,
        args.target_range,
        rows,
    )

    unmapped = [r for r in rows if r["status"] == "unmapped"]
    pending = [r for r in rows if r["status"] == "pending"]
    explained = len(rows) - len(unmapped) - len(pending)
    print(
        f"audit finished: total={len(rows)} auto={sum(r['decision_source']=='auto' for r in rows)} "
        f"manual={sum(r['decision_source']=='manual' for r in rows)} "
        f"pending={len(pending)} unmapped={len(unmapped)} explained={explained}"
    )
    print(f"ledger: {output_ledger}")
    print(f"report: {output_report}")

    if args.require_no_unmapped and unmapped:
        return 2
    if args.require_no_pending and pending:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
