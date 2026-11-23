#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Generate a CycloneDX SBOM of the third-party libraries ml-cpp ships.

The Snyk integration on this repo scans the Dockerfiles, i.e. the build images,
which says nothing about what ends up in the released zip. The libraries that do
ship (Boost, libtorch, ...) are built from source, so Snyk cannot see them.

This script turns the per-component ``3rd_party/licenses/*INFO.csv`` files (the
source of truth for shipped versions, also used by dependency_report.sh) into a
CycloneDX JSON document that ``snyk sbom test`` can analyse.

Only components Snyk can match are emitted. Snyk resolves ``pkg:pypi/...`` through
its PyPI database and some ``pkg:generic/...`` names through its unmanaged C/C++
database. A component Snyk does not recognise is silently ignored by the scan, so
those are reported on stderr instead of being emitted: a clean scan must not be
mistaken for a component having been checked.

To cover another component, add it to ``_PURL_BY_NAME`` after confirming Snyk
matches it, e.g. by testing an old, known-vulnerable version.

Usage:
    generate_sbom.py [--licenses-dir DIR] [--output FILE]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LICENSES_DIR = _REPO_ROOT / "3rd_party" / "licenses"

# INFO.csv component name (lowercased) -> (purl type and name, SBOM component name).
# Each entry has been confirmed to be matched by Snyk; see the module docstring.
_PURL_BY_NAME: Dict[str, Tuple[str, str]] = {
    "boost c++ libraries": ("generic/boost", "boost"),
    "libxml2": ("generic/libxml2", "libxml2"),
    # libtorch is built from the PyTorch sources that the PyPI "torch" package ships.
    "pytorch": ("pypi/torch", "torch"),
}


def _read_info_rows(licenses_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for info_file in sorted(licenses_dir.glob("*INFO.csv")):
        with info_file.open(newline="") as handle:
            file_rows = list(csv.DictReader(handle))
        # dependency_report.cmake only uses the last line of each file.
        if file_rows:
            rows.append(file_rows[-1])
    return rows


def build_sbom(licenses_dir: Path) -> Tuple[dict, List[str]]:
    """Return (CycloneDX document, names of components that could not be emitted)."""
    components: List[dict] = []
    skipped: List[str] = []

    for row in _read_info_rows(licenses_dir):
        name = row.get("name", "").strip()
        version = row.get("version", "").strip()
        mapping: Optional[Tuple[str, str]] = _PURL_BY_NAME.get(name.lower())
        if mapping is None or not version:
            skipped.append(name)
            continue
        purl_path, component_name = mapping
        purl = f"pkg:{purl_path}@{version}"
        components.append(
            {
                "type": "library",
                "name": component_name,
                "version": version,
                "purl": purl,
                "bom-ref": purl,
            }
        )

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {"component": {"type": "application", "name": "ml-cpp", "version": "0"}},
        "components": components,
    }
    return sbom, skipped


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--licenses-dir", type=Path, default=_DEFAULT_LICENSES_DIR)
    parser.add_argument("--output", type=Path, default=Path("sbom.cdx.json"))
    args = parser.parse_args(argv)

    sbom, skipped = build_sbom(args.licenses_dir)

    if skipped:
        print(
            "Not scanned (no Snyk-matchable purl mapping): " + ", ".join(skipped),
            file=sys.stderr,
        )
    if not sbom["components"]:
        sys.exit("error: no scannable components found in " + str(args.licenses_dir))

    args.output.write_text(json.dumps(sbom, indent=2) + "\n")
    print(
        "Wrote {} component(s) to {}: {}".format(
            len(sbom["components"]),
            args.output,
            ", ".join(c["purl"] for c in sbom["components"]),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
