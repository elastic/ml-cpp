# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Shared helpers for changelog YAML validation (schema + filename rules)."""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


ES_SCHEMA_URL = (
    "https://raw.githubusercontent.com/elastic/elasticsearch/main/"
    "build-tools-internal/src/main/resources/changelog-schema.json"
)


def load_schema(local_path: Path) -> dict:
    """Load the changelog schema, preferring the canonical ES version.

    Fetches the schema from the Elasticsearch repo to ensure we validate
    against the single source of truth. Falls back to the local copy if
    the fetch fails (e.g. no network / offline development). Warns if
    the local copy has diverged from the remote.
    """
    local_schema = None
    if local_path.exists():
        with open(local_path) as f:
            local_schema = json.load(f)

    try:
        response = urllib.request.urlopen(ES_SCHEMA_URL, timeout=10)
        remote_schema = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        if local_schema is not None:
            print(
                f"Note: could not fetch ES schema ({e}), using local copy",
                file=sys.stderr,
            )
            return local_schema
        raise RuntimeError(
            f"could not fetch ES schema and no local copy at {local_path}"
        ) from e

    if local_schema is not None and local_schema != remote_schema:
        print(
            "WARNING: local changelog-schema.json differs from the Elasticsearch source.\n"
            f"  Remote: {ES_SCHEMA_URL}\n"
            f"  Local:  {local_path}\n"
            "  Validating against the remote (canonical) schema.\n"
            "  Please update the local copy to stay in sync.\n",
            file=sys.stderr,
        )

    return remote_schema


def filename_convention_errors(filename: str, stem: str, data: dict) -> list[str]:
    """Filename / ``pr`` field consistency (same rules as validate_changelogs)."""
    errors: list[str] = []
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


def schema_validation_errors(data: dict, schema: dict, filename: str) -> list[str]:
    """JSON Schema validation errors for a parsed changelog mapping."""
    import jsonschema

    errors: list[str] = []
    validator = jsonschema.Draft7Validator(schema)
    for error in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in error.absolute_path) or "(root)"
        errors.append(f"{filename}: {path}: {error.message}")
    return errors


def validate_changelog_mapping(filename: str, stem: str, data: dict, schema: dict) -> list[str]:
    """Validate a single parsed YAML document (mapping) for export / tooling."""
    errors = schema_validation_errors(data, schema, filename)
    errors.extend(filename_convention_errors(filename, stem, data))
    return errors
