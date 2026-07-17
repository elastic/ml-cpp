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

    # If an export branch already exists locally or on origin, a timestamp suffix
    # is used automatically; --force-branch deletes a colliding local branch first.
    python3 dev-tools/export_changelogs.py --target ~/src/elasticsearch/docs/changelog --create-pr --force-branch
"""

import argparse
import difflib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from changelog_common import load_schema, validate_changelog_mapping

try:
    import yaml
except ImportError:
    print("Missing pyyaml. Install with: pip3 install pyyaml", file=sys.stderr)
    sys.exit(2)

# fail fast with a clear message if jsonschema isn’t installed
try:
    import jsonschema  # noqa: F401
except ImportError:
    print("Missing jsonschema. Install with: pip3 install jsonschema", file=sys.stderr)
    sys.exit(2)

PREFIX = "ml-cpp-"
SOURCE_REPO = "elastic/ml-cpp"


def validate_entries(entries, schema: dict) -> list[str]:
    """Validate all entries (schema + filename rules). Returns list of errors."""
    errors: list[str] = []
    for source_path, _, data in entries:
        errors.extend(
            validate_changelog_mapping(source_path.name, source_path.stem, data, schema)
        )
    return errors


def pick_export_branch_name(es_repo: Path, base_name: str, force_branch: bool) -> str:
    """Return a branch name that does not collide with an existing local or origin branch."""
    if force_branch:
        subprocess.run(
            ["git", "branch", "-D", base_name],
            cwd=es_repo,
            capture_output=True,
            text=True,
        )
        if not _remote_branch_exists(es_repo, base_name):
            return base_name

    if not _local_branch_exists(es_repo, base_name) and not _remote_branch_exists(
        es_repo, base_name
    ):
        return base_name

    suffix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    return f"{base_name}-{suffix}"


def _local_branch_exists(es_repo: Path, name: str) -> bool:
    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{name}"],
        cwd=es_repo,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _remote_branch_exists(es_repo: Path, name: str) -> bool:
    result = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", f"refs/heads/{name}"],
        cwd=es_repo,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


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


def resolve_conflict(source_path, dest, target_name):
    """Handle a pre-existing file at the destination. Returns the action taken."""
    source_lines = source_path.read_text().splitlines(keepends=True)
    dest_lines = dest.read_text().splitlines(keepends=True)

    if source_lines == dest_lines:
        print(f"  {target_name}: identical to existing file, skipping")
        return "skip"

    print(f"\n  {target_name}: file already exists with different content.\n")
    diff = difflib.unified_diff(
        dest_lines, source_lines,
        fromfile=f"existing: {dest.name}",
        tofile=f"incoming: {source_path.name}",
    )
    sys.stdout.writelines("    " + line for line in diff)
    print()

    while True:
        choice = input(f"  [{target_name}] (o)verwrite / (s)kip / (a)bort export? ").strip().lower()
        if choice in ("o", "overwrite"):
            write_entry_with_source_repo(source_path, dest)
            print(f"  {target_name}: overwritten")
            return "overwrite"
        elif choice in ("s", "skip"):
            print(f"  {target_name}: skipped")
            return "skip"
        elif choice in ("a", "abort"):
            print("\nExport aborted.")
            sys.exit(1)
        else:
            print("  Please enter 'o' (overwrite), 's' (skip), or 'a' (abort).")


def verify_es_repo(target_dir):
    """Verify that the target looks like an ES docs/changelog directory."""
    target = Path(target_dir).resolve()

    if not target.is_dir():
        print(f"Error: target directory does not exist: {target}", file=sys.stderr)
        sys.exit(1)

    es_repo_root = target.parent.parent
    markers = [
        es_repo_root / "build.gradle",
        es_repo_root / "settings.gradle",
        es_repo_root / "docs" / "changelog",
    ]
    if not all(m.exists() for m in markers):
        print(
            f"Warning: {es_repo_root} does not look like an Elasticsearch checkout.\n"
            f"  Expected to find build.gradle, settings.gradle, and docs/changelog/\n"
            f"  at the repo root (two levels above --target).\n",
            file=sys.stderr,
        )
        choice = input("  Continue anyway? (y/n) ").strip().lower()
        if choice not in ("y", "yes"):
            print("Export aborted.")
            sys.exit(1)

    return es_repo_root


def write_entry_with_source_repo(source_path, dest):
    """Write a changelog entry to dest, injecting source_repo if not already present."""
    with open(source_path) as f:
        data = yaml.safe_load(f)
    if "source_repo" not in data:
        data["source_repo"] = SOURCE_REPO
    with open(dest, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def export_entries(entries, target_dir, dry_run=False):
    """Export entries to the target directory with prefixed filenames and source_repo.

    Returns (exported_dest_paths, exported_source_paths). The second list is only
    populated for files actually written or overwritten from ml-cpp (used by
    :option:`--prune`).
    """
    target = Path(target_dir)

    exported_dests: list[Path] = []
    exported_sources: list[Path] = []
    skipped = 0
    for source_path, target_name, data in entries:
        dest = target / target_name
        pr = data.get("pr", "n/a")
        summary = data.get("summary", "")[:60]
        if dry_run:
            flag = " [EXISTS]" if dest.exists() else ""
            print(f"  {target_name}  (PR #{pr}: {summary}){flag}")
            exported_dests.append(dest)
        elif dest.exists():
            action = resolve_conflict(source_path, dest, target_name)
            if action == "overwrite":
                exported_dests.append(dest)
                exported_sources.append(source_path)
            else:
                skipped += 1
        else:
            write_entry_with_source_repo(source_path, dest)
            print(f"  Copied {source_path.name} -> {target_name}")
            exported_dests.append(dest)
            exported_sources.append(source_path)

    if skipped > 0 and not dry_run:
        print(f"\n  ({skipped} file(s) skipped due to conflicts)")

    return exported_dests, exported_sources


def create_pr(es_repo_dir, exported_files, version=None, force_branch: bool = False):
    """Create a git branch and PR in the ES repo with the exported entries."""
    es_repo = Path(es_repo_dir).resolve()
    base_branch = "ml-cpp-changelog-export"
    if version:
        base_branch += f"-{version}"
    branch_name = pick_export_branch_name(es_repo, base_branch, force_branch)
    if branch_name != base_branch:
        print(
            f"Note: branch '{base_branch}' already exists; using '{branch_name}' instead.",
            file=sys.stderr,
        )

    try:
        subprocess.run(["git", "checkout", "-b", branch_name], cwd=es_repo, check=True)
        subprocess.run(["git", "add"] + [str(f) for f in exported_files], cwd=es_repo, check=True)

        msg = "[ML] Add ml-cpp changelog entries"
        if version:
            msg += f" for {version}"
        subprocess.run(["git", "commit", "-m", msg], cwd=es_repo, check=True)
        subprocess.run(["git", "push", "-u", "origin", branch_name], cwd=es_repo, check=True)

        pr_body = (
            "Adds ml-cpp changelog entries to the ES release notes.\n\n"
            "Source: elastic/ml-cpp docs/changelog/"
        )
        if version:
            pr_body += f"\nVersion: {version}"
        result = subprocess.run(
            ["gh", "pr", "create", "--title", msg, "--body", pr_body],
            cwd=es_repo, capture_output=True, text=True,
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
        description="Export ml-cpp changelog entries for ES release notes",
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
        "--force-branch",
        action="store_true",
        help="When using --create-pr, delete a local branch with the same name if it exists "
        "(still uses a timestamp suffix if the branch already exists on origin)",
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
    schema_path = repo_root / "docs" / "changelog" / "changelog-schema.json"

    entries = collect_entries(changelog_dir, args.files if args.files else None)
    if not entries:
        print("No changelog entries found.")
        return

    print(f"Found {len(entries)} changelog entry(ies).")

    # Validate all entries before exporting (same rules as validate_changelogs.py)
    try:
        print("Validating entries against schema...", flush=True)
        schema = load_schema(schema_path)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    errors = validate_entries(entries, schema)
    if errors:
        print(f"Validation FAILED ({len(errors)} error(s)):\n")
        for error in errors:
            print(f"  - {error}")
        print("\nFix validation errors before exporting.")
        sys.exit(1)
    print("OK")

    print()

    if args.dry_run or not args.target:
        export_entries(entries, args.target or "/dev/null", dry_run=True)
        if not args.target:
            print("\nUse --target to export, or --dry-run to preview.")
        return

    # Verify the target is a real ES checkout
    es_repo_root = verify_es_repo(args.target)

    exported_dests, exported_sources = export_entries(entries, args.target)
    if not exported_dests:
        print("\nNo files exported.")
        return

    print(f"\nExported {len(exported_dests)} file(s) to {args.target}")

    if args.create_pr:
        create_pr(es_repo_root, exported_dests, args.version, force_branch=args.force_branch)

    if args.prune:
        for source_path in exported_sources:
            source_path.unlink()
            print(f"  Pruned {source_path}")
        print(f"\nPruned {len(exported_sources)} source file(s)")


if __name__ == "__main__":
    main()
