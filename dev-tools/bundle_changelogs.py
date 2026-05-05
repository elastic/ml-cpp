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
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Missing pyyaml. Install with: pip3 install pyyaml", file=sys.stderr)
    sys.exit(2)


TYPE_ORDER = [
    ("known-issue", "Known issues"),
    ("security", "Security fixes"),
    ("breaking", "Breaking changes"),
    ("breaking-java", "Breaking Java changes"),
    ("deprecation", "Deprecations"),
    ("feature", "New features"),
    ("new-aggregation", "New aggregations"),
    ("enhancement", "Enhancements"),
    ("bug", "Bug fixes"),
    ("regression", "Regression fixes"),
    ("upgrade", "Upgrades"),
]

ML_CPP_PULL_URL = "https://github.com/elastic/ml-cpp/pull"
ML_CPP_ISSUE_URL = "https://github.com/elastic/ml-cpp/issues"


@dataclass(frozen=True)
class ChangelogFormatStyle:
    """Heading / bullet / link conventions for bundled changelog output."""

    version_heading_prefix: str
    type_heading_prefix: str
    use_markdown_issue_links: bool

    def version_heading(self, version: str) -> str:
        return f"{self.version_heading_prefix}{version}\n"

    def type_heading(self, label: str) -> str:
        return f"{self.type_heading_prefix}{label}\n"

    def area_line(self, area: str) -> str:
        if self.use_markdown_issue_links:
            return f"**{area}**"
        return f"*{area}*"

    def entry_line(self, entry: dict) -> str:
        summary = entry["summary"]
        issues = entry.get("issues", [])
        pr = entry.get("pr")
        if self.use_markdown_issue_links:
            issue_refs = ", ".join(f"#{i}" for i in issues)
            if pr:
                line = f"- {summary} [#{pr}]({ML_CPP_PULL_URL}/{pr})"
            else:
                line = f"- {summary}"
        else:
            issue_refs = ", ".join(
                f"{ML_CPP_ISSUE_URL}/{i}[#{i}]" for i in issues
            )
            if pr:
                line = f"* {summary} {{ml-pull}}{pr}[#{pr}]"
            else:
                line = f"* {summary}"
        if issue_refs:
            line += f" ({issue_refs})"
        return line


MARKDOWN_STYLE = ChangelogFormatStyle("## ", "### ", True)
ASCIIDOC_STYLE = ChangelogFormatStyle("== ", "=== ", False)


def load_entries(changelog_dir):
    entries = []
    for path in sorted(changelog_dir.glob("*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)
            if data and isinstance(data, dict):
                data["_file"] = path.name
                entries.append(data)
    return entries


def _group_by_type_and_area(entries):
    grouped = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        area = entry.get("area", "General")
        grouped[entry["type"]][area].append(entry)
    return grouped


def format_entries(entries, style: ChangelogFormatStyle, version=None):
    """Render bundled entries using the given heading/bullet/link style."""
    lines = []
    if version:
        lines.append(style.version_heading(version))

    grouped = _group_by_type_and_area(entries)

    for type_key, type_label in TYPE_ORDER:
        if type_key not in grouped:
            continue
        lines.append(style.type_heading(type_label))
        for area in sorted(grouped[type_key].keys()):
            lines.append(style.area_line(area))
            for entry in sorted(grouped[type_key][area], key=lambda e: e.get("pr", 0)):
                lines.append(style.entry_line(entry))
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

    style = ASCIIDOC_STYLE if args.format == "asciidoc" else MARKDOWN_STYLE
    print(format_entries(entries, style, args.version))


if __name__ == "__main__":
    main()
