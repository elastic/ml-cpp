#!/usr/bin/env python3
"""
Export ml-cpp changelog entries for inclusion in Elasticsearch release notes.

Copies changelog YAML files from docs/changelog/ to a target directory
(typically elastic/elasticsearch's docs/changelog/) with a 'ml-cpp-' filename
prefix to avoid PR number collisions with ES-native entries.

Usage:
    # Preview what would be exported
    python3 dev-tools/export_changelogs.py --dry-run

    # Export to a local ES checkout
    python3 dev-tools/export_changelogs.py --target ~/src/elasticsearch/docs/changelog

    # Export and create a PR in the ES repo
    python3 dev-tools/export_changelogs.py --target ~/src/elasticsearch/docs/changelog --create-pr

    # Export specific files only
    python3 dev-tools/export_changelogs.py --target /tmp/out docs/changelog/3008.yaml
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Missing pyyaml. Install with: pip3 install pyyaml", file=sys.stderr)
    sys.exit(2)


PREFIX = "ml-cpp-"


def collect_entries(changelog_dir, specific_files=None):
    """Collect changelog YAML files, returning (source_path, target_name, data) tuples."""
    if specific_files:
        paths = [Path(f) for f in specific_files]
    else:
        paths = sorted(changelog_dir.glob("*.yaml"))

    entries = []
    for path in paths:
        if not path.exists():
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data or not isinstance(data, dict):
            continue

        target_name = PREFIX + path.name
        entries.append((path, target_name, data))

    return entries


def export_entries(entries, target_dir, dry_run=False):
    """Copy entries to the target directory with prefixed filenames."""
    target = Path(target_dir)
    if not dry_run and not target.is_dir():
        print(f"Error: target directory {target} does not exist", file=sys.stderr)
        sys.exit(1)

    for source_path, target_name, data in entries:
        dest = target / target_name
        pr = data.get("pr", "n/a")
        summary = data.get("summary", "")[:60]
        if dry_run:
            print(f"  {target_name}  (PR #{pr}: {summary})")
        else:
            shutil.copy2(source_path, dest)
            print(f"  Copied {source_path.name} -> {dest}")

    return [target / name for _, name, _ in entries]


def create_pr(es_repo_dir, exported_files, version=None):
    """Create a git branch and PR in the ES repo with the exported entries."""
    es_repo = Path(es_repo_dir).resolve()
    branch_name = f"ml-cpp-changelog-export"
    if version:
        branch_name += f"-{version}"

    try:
        subprocess.run(["git", "checkout", "-b", branch_name], cwd=es_repo, check=True)
        subprocess.run(["git", "add"] + [str(f) for f in exported_files], cwd=es_repo, check=True)

        msg = "[ML] Add ml-cpp changelog entries"
        if version:
            msg += f" for {version}"
        subprocess.run(["git", "commit", "-m", msg], cwd=es_repo, check=True)
        subprocess.run(["git", "push", "-u", "origin", branch_name], cwd=es_repo, check=True)

        pr_body = f"Adds ml-cpp changelog entries to the ES release notes.\n\nSource: elastic/ml-cpp docs/changelog/"
        if version:
            pr_body += f"\nVersion: {version}"
        result = subprocess.run(
            ["gh", "pr", "create", "--title", msg, "--body", pr_body],
            cwd=es_repo, capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"\nPR created: {result.stdout.strip()}")
        else:
            print(f"\nFailed to create PR: {result.stderr}", file=sys.stderr)
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Git error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Export ml-cpp changelog entries for ES release notes"
    )
    parser.add_argument(
        "--target",
        help="Target directory (e.g. ~/src/elasticsearch/docs/changelog)",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Source changelog directory (default: docs/changelog/)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version label (used in PR title/branch if --create-pr)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported without copying files",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a PR in the ES repo (requires --target to be inside an ES checkout)",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete source YAML files after successful export (use after release)",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific changelog files to export (default: all *.yaml in --dir)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    changelog_dir = Path(args.dir) if args.dir else repo_root / "docs" / "changelog"

    entries = collect_entries(changelog_dir, args.files if args.files else None)
    if not entries:
        print("No changelog entries found.")
        return

    print(f"Found {len(entries)} changelog entry(ies):\n")

    if args.dry_run or not args.target:
        export_entries(entries, args.target or "/dev/null", dry_run=True)
        if not args.target:
            print("\nUse --target to export, or --dry-run to preview.")
        return

    exported = export_entries(entries, args.target)
    print(f"\nExported {len(exported)} file(s) to {args.target}")

    if args.create_pr:
        es_repo_dir = Path(args.target).resolve().parent.parent
        create_pr(es_repo_dir, exported, args.version)

    if args.prune:
        for source_path, _, _ in entries:
            source_path.unlink()
            print(f"  Pruned {source_path}")
        print(f"\nPruned {len(entries)} source file(s)")


if __name__ == "__main__":
    main()
