#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Pytest tests for dev-tools/generate_sbom.py.

Verifies the CycloneDX SBOM built from 3rd_party/licenses/*INFO.csv only lists
components that Snyk can match, and that components it cannot match are reported
rather than silently dropped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_DEV_TOOLS = Path(__file__).resolve().parents[1]
if str(_DEV_TOOLS) not in sys.path:
    sys.path.insert(0, str(_DEV_TOOLS))

import generate_sbom as gs  # noqa: E402

_REPO_ROOT = _DEV_TOOLS.parent
_LICENSES = _REPO_ROOT / "3rd_party" / "licenses"

_HEADER = "name,version,revision,url,license,copyright,sourceURL\n"


def _write_info(directory: Path, filename: str, row: str) -> None:
    (directory / filename).write_text(_HEADER + row + "\n")


def test_mapped_component_becomes_purl_component(tmp_path: Path) -> None:
    _write_info(tmp_path, "boost-INFO.csv", "Boost C++ Libraries,1.86.0,,http://www.boost.org,BSL-1.0,,")

    sbom, skipped = gs.build_sbom(tmp_path)

    assert skipped == []
    [component] = sbom["components"]
    assert component["name"] == "boost"
    assert component["version"] == "1.86.0"
    assert component["purl"] == "pkg:generic/boost@1.86.0"


def test_pytorch_maps_to_pypi_torch(tmp_path: Path) -> None:
    _write_info(tmp_path, "pytorch-INFO.csv", "PyTorch,2.7.1,abc123,https://pytorch.org,BSD-3-Clause,,")

    sbom, _ = gs.build_sbom(tmp_path)

    [component] = sbom["components"]
    assert component["purl"] == "pkg:pypi/torch@2.7.1"


def test_unmapped_component_is_reported_not_emitted(tmp_path: Path) -> None:
    _write_info(tmp_path, "scowl-INFO.csv", "SCOWL,7.1,,http://wordlist.aspell.net,Custom,,")

    sbom, skipped = gs.build_sbom(tmp_path)

    assert sbom["components"] == []
    assert skipped == ["SCOWL"]


def test_sbom_is_valid_cyclonedx_envelope(tmp_path: Path) -> None:
    _write_info(tmp_path, "boost-INFO.csv", "Boost C++ Libraries,1.86.0,,http://www.boost.org,BSL-1.0,,")

    sbom, _ = gs.build_sbom(tmp_path)

    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.5"
    assert sbom["metadata"]["component"]["name"] == "ml-cpp"
    # Snyk requires every component to carry a bom-ref and a purl.
    for component in sbom["components"]:
        assert component["bom-ref"]
        assert component["purl"].startswith("pkg:")


def test_component_with_empty_version_is_reported(tmp_path: Path) -> None:
    _write_info(tmp_path, "boost-INFO.csv", "Boost C++ Libraries,,,http://www.boost.org,BSL-1.0,,")

    sbom, skipped = gs.build_sbom(tmp_path)

    assert sbom["components"] == []
    assert skipped == ["Boost C++ Libraries"]


def test_real_license_dir_yields_boost_and_torch() -> None:
    sbom, _ = gs.build_sbom(_LICENSES)

    purls = {c["purl"] for c in sbom["components"]}
    assert any(p.startswith("pkg:generic/boost@") for p in purls)
    assert any(p.startswith("pkg:pypi/torch@") for p in purls)


def test_main_writes_json_file(tmp_path: Path) -> None:
    out = tmp_path / "sbom.cdx.json"

    rc = gs.main(["--licenses-dir", str(_LICENSES), "--output", str(out)])

    assert rc == 0
    assert json.loads(out.read_text())["bomFormat"] == "CycloneDX"


def test_main_fails_when_nothing_is_scannable(tmp_path: Path) -> None:
    empty = tmp_path / "licenses"
    empty.mkdir()

    with pytest.raises(SystemExit) as exc:
        gs.main(["--licenses-dir", str(empty), "--output", str(tmp_path / "out.json")])

    assert exc.value.code != 0
