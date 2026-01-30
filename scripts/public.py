#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


def repo_root() -> Path:
    # Resolve the repository root based on this script's location.
    return Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    # Load the JSON release configuration from disk.
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_gitmodule_paths(path: Path) -> set:
    # Collect all submodule paths declared in a .gitmodules file.
    paths = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*path\s*=\s*(.+)\s*", line)
        if match:
            paths.add(match.group(1))
    return paths


def rewrite_gitmodules(path: Path, url_by_path: dict, dry_run: bool) -> set:
    # Rewrite submodule URLs in .gitmodules based on submodule path keys.
    if not url_by_path:
        return set()
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    current_path = None
    updated = set()
    out = []
    for line in lines:
        match = re.match(r'\s*\[submodule\s+"([^"]+)"\]\s*', line)
        if match:
            current_path = None
            out.append(line)
            continue
        match = re.match(r"\s*path\s*=\s*(.+)\s*", line)
        if match:
            current_path = match.group(1)
            out.append(line)
            continue
        match = re.match(r"(\s*url\s*=\s*)(.+)\s*", line)
        if match and current_path in url_by_path:
            prefix = match.group(1)
            out.append(f"{prefix}{url_by_path[current_path]}\n")
            updated.add(current_path)
            continue
        out.append(line)
    if not dry_run:
        path.write_text("".join(out), encoding="utf-8")
    return updated


def remove_path(path: Path, dry_run: bool) -> None:
    # Delete a file or directory, with dry-run support.
    if dry_run:
        print(f"[dry-run] remove: {path}")
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"[remove] {path}")


def run_git(args: list[str], cwd: Path) -> None:
    # Run a git command and raise on failure.
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def ensure_branch(branch: str, cwd: Path) -> None:
    # Switch to a local branch, creating it if it doesn't exist.
    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=str(cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        run_git(["checkout", "-b", branch], cwd)
    else:
        run_git(["checkout", branch], cwd)


def update_submodule_remotes(root: Path, url_by_path: dict, remote_name: str, dry_run: bool) -> set:
    # Ensure each submodule has the target remote pointing at the public URL.
    updated = set()
    for rel_path, url in url_by_path.items():
        submodule_path = (root / rel_path).resolve()
        assert submodule_path.exists(), f"Missing submodule path: {submodule_path}"
        if dry_run:
            print(f"[dry-run] set-remote: {rel_path} {remote_name} -> {url}")
            continue
        result = subprocess.run(
            ["git", "-C", str(submodule_path), "remote", "get-url", remote_name],
            cwd=str(root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            run_git(["-C", str(submodule_path), "remote", "add", remote_name, url], root)
        else:
            run_git(
                ["-C", str(submodule_path), "remote", "set-url", remote_name, url],
                root,
            )
        print(f"[update] {rel_path}: {remote_name} -> {url}")
        updated.add(rel_path)
    return updated


def update_submodule_refs(root: Path, ref_by_path: dict, remote_name: str, dry_run: bool) -> set:
    # Checkout target refs in submodules to update gitlink commits.
    updated = set()
    for rel_path, ref in ref_by_path.items():
        submodule_path = (root / rel_path).resolve()
        assert submodule_path.exists(), f"Missing submodule path: {submodule_path}"
        assert ref and ref == ref.strip(), f"Invalid ref for {rel_path}"
        if dry_run:
            print(f"[dry-run] checkout: {rel_path} -> {ref}")
            continue
        has_ref = subprocess.run(
            [
                "git",
                "-C",
                str(submodule_path),
                "rev-parse",
                "--verify",
                "--quiet",
                f"{ref}^{{commit}}",
            ],
            cwd=str(root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if has_ref.returncode != 0:
            run_git(
                [
                    "-C",
                    str(submodule_path),
                    "fetch",
                    "--no-tags",
                    "--depth",
                    "1",
                    remote_name,
                    ref,
                ],
                root,
            )
        run_git(["-C", str(submodule_path), "checkout", ref], root)
        print(f"[update] {rel_path}: checkout -> {ref}")
        updated.add(rel_path)
    return updated


def main() -> int:
    # Parse CLI arguments and load configuration.
    parser = argparse.ArgumentParser(
        description="Prepare the repo for public release by rewriting submodule URLs and removing internal files."
    )
    parser.add_argument(
        "--config",
        default=str(repo_root() / "scripts" / "public.json"),
        help="Path to release config JSON.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
    args = parser.parse_args()

    # Resolve repo/config paths and enforce release branch when not dry-running.
    root = repo_root()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if not args.dry_run:
        ensure_branch("github-public", root)

    # Read config sections.
    gitmodules_updates = config.get("gitmodule_updates", {})
    remove_paths = config.get("remove_paths", [])
    gitmodule_refs = config.get("gitmodule_refs", {})

    # Validate submodule paths and build URL mappings.
    gitmodules_path = (root / ".gitmodules").resolve()
    assert gitmodules_path.exists(), f"Missing .gitmodules: {gitmodules_path}"

    paths_in_gitmodules = parse_gitmodule_paths(gitmodules_path)

    url_by_path = {}
    for submodule_path, url in gitmodules_updates.items():
        assert url and url == url.strip(), f"Invalid URL for {submodule_path}"
        assert (
            submodule_path
            in paths_in_gitmodules), f"Submodule path not found in .gitmodules: {submodule_path}"
        url_by_path[submodule_path] = url

    # Rewrite .gitmodules URLs.
    updated = rewrite_gitmodules(gitmodules_path, url_by_path, args.dry_run)
    assert updated == set(url_by_path.keys()), "Not all submodules were updated"

    for name in sorted(updated):
        print(f"[update] .gitmodules: {name}")

    # Validate submodule refs target declared paths.
    for submodule_path in gitmodule_refs.keys():
        assert (
            submodule_path
            in paths_in_gitmodules), f"Submodule path not found in .gitmodules: {submodule_path}"

    # Update submodule remotes and gitlink commits.
    remote_name = "github"
    update_submodule_remotes(root, url_by_path, remote_name, args.dry_run)
    updated_submodules = update_submodule_refs(root, gitmodule_refs, remote_name, args.dry_run)

    # Remove internal-only files.
    for rel_path in remove_paths:
        path = (root / rel_path).resolve()
        assert path.exists(), f"Missing remove path: {path}"
        remove_path(path, args.dry_run)

    # Stage only release-related changes.
    if args.dry_run:
        print("Dry-run complete.")
    else:
        stage_paths = [".gitmodules", *remove_paths, *sorted(updated_submodules)]
        stage_paths = list(dict.fromkeys(stage_paths))
        run_git(["add", "-A", "--", *stage_paths], root)
        print("[update] staged changes")
        print("Release prep complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
