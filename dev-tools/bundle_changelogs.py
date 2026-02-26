#!/usr/bin/env python3
"""
Bundle per-PR changelog YAML files into a consolidated changelog for release.

Usage:
    python3 bundle_changelogs.py [--dir DIR] [--version VERSION] [--format FORMAT]

Outputs a formatted changelog grouped by type and area, suitable for inclusion
in release notes.

Formats:
    markdown (default) - Markdown suitable for GitHub releases
    asciidoc          - AsciiDoc suitable for Elastic docs
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Missing pyyaml. Install with: pip3 install pyyaml", file=sys.stderr)
    sys.exit(2)


TYPE_ORDER = [
    ("breaking", "Breaking changes"),
    ("deprecation", "Deprecations"),
    ("feature", "New features"),
    ("enhancement", "Enhancements"),
    ("bug", "Bug fixes"),
    ("regression", "Regression fixes"),
]


def load_entries(changelog_dir):
    entries = []
    for path in sorted(changelog_dir.glob("*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)
            if data and isinstance(data, dict):
                data["_file"] = path.name
                entries.append(data)
    return entries


def format_markdown(entries, version=None):
    lines = []
    if version:
        lines.append(f"## {version}\n")

    grouped = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        grouped[entry["type"]][entry["area"]].append(entry)

    for type_key, type_label in TYPE_ORDER:
        if type_key not in grouped:
            continue
        lines.append(f"### {type_label}\n")
        for area in sorted(grouped[type_key].keys()):
            lines.append(f"**{area}**")
            for entry in sorted(grouped[type_key][area], key=lambda e: e["pr"]):
                pr = entry["pr"]
                summary = entry["summary"]
                issues = entry.get("issues", [])
                issue_refs = ", ".join(f"#{i}" for i in issues)
                line = f"- {summary} [#{pr}](https://github.com/elastic/ml-cpp/pull/{pr})"
                if issue_refs:
                    line += f" ({issue_refs})"
                lines.append(line)
            lines.append("")

    return "\n".join(lines)


def format_asciidoc(entries, version=None):
    lines = []
    if version:
        lines.append(f"== {version}\n")

    grouped = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        grouped[entry["type"]][entry["area"]].append(entry)

    for type_key, type_label in TYPE_ORDER:
        if type_key not in grouped:
            continue
        lines.append(f"=== {type_label}\n")
        for area in sorted(grouped[type_key].keys()):
            lines.append(f"*{area}*")
            for entry in sorted(grouped[type_key][area], key=lambda e: e["pr"]):
                pr = entry["pr"]
                summary = entry["summary"]
                issues = entry.get("issues", [])
                issue_refs = ", ".join(f"https://github.com/elastic/ml-cpp/issues/{i}[#{i}]" for i in issues)
                line = f"* {summary} https://github.com/elastic/ml-cpp/pull/{pr}[#{pr}]"
                if issue_refs:
                    line += f" ({issue_refs})"
                lines.append(line)
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Bundle changelog YAML files")
    parser.add_argument("--dir", default=None, help="Changelog directory")
    parser.add_argument("--version", default=None, help="Version string for heading")
    parser.add_argument("--format", default="markdown", choices=["markdown", "asciidoc"])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    changelog_dir = Path(args.dir) if args.dir else repo_root / "docs" / "changelog"

    entries = load_entries(changelog_dir)
    if not entries:
        print("No changelog entries found.", file=sys.stderr)
        sys.exit(0)

    if args.format == "asciidoc":
        print(format_asciidoc(entries, args.version))
    else:
        print(format_markdown(entries, args.version))


if __name__ == "__main__":
    main()
