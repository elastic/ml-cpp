#!/usr/bin/env python3
"""
Validate changelog YAML files against the changelog JSON schema.

Usage:
    python3 validate_changelogs.py [--schema SCHEMA] [--dir DIR] [FILES...]

If FILES are given, only those files are validated.
Otherwise all *.yaml files in DIR (default: docs/changelog/) are validated.

Exit codes:
    0  All files valid (or no files to validate)
    1  One or more validation errors
    2  Missing dependencies or bad arguments
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def check_dependencies():
    """Check that required Python packages are available."""
    missing = []
    try:
        import yaml  # noqa: F401
    except ImportError:
        missing.append("pyyaml")
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        missing.append("jsonschema")
    if missing:
        print(
            f"Missing Python packages: {', '.join(missing)}\n"
            f"Install with: pip3 install {' '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(2)


def load_schema(schema_path):
    with open(schema_path) as f:
        return json.load(f)


def validate_file(filepath, schema):
    """Validate a single YAML file. Returns a list of error strings."""
    import jsonschema
    import yaml

    errors = []
    filename = os.path.basename(filepath)
    stem = Path(filepath).stem

    try:
        with open(filepath) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        errors.append(f"{filename}: invalid YAML: {e}")
        return errors

    if data is None:
        errors.append(f"{filename}: file is empty")
        return errors

    if not isinstance(data, dict):
        errors.append(f"{filename}: expected a YAML mapping, got {type(data).__name__}")
        return errors

    # Validate against JSON schema
    validator = jsonschema.Draft7Validator(schema)
    for error in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in error.absolute_path) or "(root)"
        errors.append(f"{filename}: {path}: {error.message}")

    # Filename convention: numeric filenames must match the pr field.
    # Types without a pr field (known-issue, security) may use descriptive names.
    if re.match(r"^\d+$", stem):
        if "pr" in data and data["pr"] != int(stem):
            errors.append(
                f"{filename}: pr field ({data['pr']}) does not match filename ({stem})"
            )
    elif "pr" in data:
        errors.append(
            f"{filename}: file has a pr field ({data['pr']}), "
            f"so filename should be {data['pr']}.yaml"
        )

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate changelog YAML files")
    parser.add_argument(
        "--schema",
        default=None,
        help="Path to the JSON schema (default: docs/changelog/changelog-schema.json)",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Directory containing changelog YAML files (default: docs/changelog/)",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to validate (overrides --dir)",
    )
    args = parser.parse_args()

    check_dependencies()

    # Resolve paths relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    schema_path = Path(args.schema) if args.schema else repo_root / "docs" / "changelog" / "changelog-schema.json"
    changelog_dir = Path(args.dir) if args.dir else repo_root / "docs" / "changelog"

    if not schema_path.exists():
        print(f"Schema not found: {schema_path}", file=sys.stderr)
        sys.exit(2)

    schema = load_schema(schema_path)

    # Collect files to validate
    if args.files:
        yaml_files = [Path(f) for f in args.files]
    else:
        yaml_files = sorted(changelog_dir.glob("*.yaml"))

    if not yaml_files:
        print("No changelog files to validate.")
        return

    all_errors = []
    for filepath in yaml_files:
        if not filepath.exists():
            all_errors.append(f"{filepath}: file not found")
            continue
        errors = validate_file(filepath, schema)
        all_errors.extend(errors)

    if all_errors:
        print(f"Changelog validation failed ({len(all_errors)} error(s)):\n")
        for error in all_errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print(f"Validated {len(yaml_files)} changelog file(s) successfully.")


if __name__ == "__main__":
    main()
