# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Unit tests for changelog YAML tooling (validate / export / bundle)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_DEV_TOOLS = Path(__file__).resolve().parent.parent
if str(_DEV_TOOLS) not in sys.path:
    sys.path.insert(0, str(_DEV_TOOLS))

from bundle_changelogs import (  # noqa: E402
    ASCIIDOC_STYLE,
    MARKDOWN_STYLE,
    format_entries,
)
from changelog_common import (  # noqa: E402
    filename_convention_errors,
    validate_changelog_mapping,
)


MINIMAL_SCHEMA = {
    "type": "object",
    "required": ["type", "summary"],
    "properties": {
        "type": {"type": "string"},
        "summary": {"type": "string"},
        "pr": {"type": "integer"},
        "issues": {"type": "array", "items": {"type": "integer"}},
    },
}


class FilenameConventionTests(unittest.TestCase):
    def test_numeric_stem_must_match_pr(self):
        data = {"type": "bug", "summary": "x", "pr": 42}
        errors = filename_convention_errors("3008.yaml", "3008", data)
        self.assertTrue(any("does not match filename" in e for e in errors))

    def test_numeric_stem_matches_pr_ok(self):
        data = {"type": "bug", "summary": "x", "pr": 3008}
        self.assertEqual(filename_convention_errors("3008.yaml", "3008", data), [])

    def test_descriptive_stem_must_not_have_pr_field(self):
        data = {"type": "known-issue", "summary": "x", "pr": 1}
        errors = filename_convention_errors("foo.yaml", "foo", data)
        self.assertTrue(any("filename should be" in e for e in errors))


class ValidateChangelogMappingTests(unittest.TestCase):
    def test_schema_and_filename_combined(self):
        data = {"type": "bug", "summary": "fix", "pr": 9}
        errors = validate_changelog_mapping("1.yaml", "1", data, MINIMAL_SCHEMA)
        self.assertTrue(any("does not match filename" in e for e in errors))


class BundleFormatTests(unittest.TestCase):
    def test_markdown_pr_link(self):
        entries = [
            {
                "type": "bug",
                "area": "Machine Learning",
                "summary": "Fix thing",
                "pr": 99,
                "issues": [100],
            }
        ]
        out = format_entries(entries, MARKDOWN_STYLE, version="9.5.0")
        self.assertIn("## 9.5.0", out)
        self.assertIn("### Bug fixes", out)
        self.assertIn("**Machine Learning**", out)
        self.assertIn("https://github.com/elastic/ml-cpp/pull/99", out)
        self.assertIn("#100", out)

    def test_asciidoc_ml_pull_macro(self):
        entries = [
            {
                "type": "enhancement",
                "area": "Machine Learning",
                "summary": "Improve",
                "pr": 7,
                "issues": [],
            }
        ]
        out = format_entries(entries, ASCIIDOC_STYLE)
        self.assertIn("=== Enhancements", out)
        self.assertIn("{ml-pull}7[#7]", out)


if __name__ == "__main__":
    unittest.main()
