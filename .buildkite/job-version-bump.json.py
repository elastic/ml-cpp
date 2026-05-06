#!/usr/bin/env python
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
# Phase 1 of the ml-cpp version bump pipeline (dynamic upload from release-eng).
#
# Buildkite step `if` expressions cannot use build meta-data (see
# https://buildkite.com/docs/pipelines/conditionals ). validate_version_bump_params.sh
# sets ml_cpp_version_bump_noop when origin already matches NEW_VERSION; phase 2
# (Slack, bump, DRA wait) is uploaded only when needed by
# dev-tools/version_bump_upload_phase2.sh.

import contextlib
import json


WOLFI_IMAGE = "docker.elastic.co/release-eng/wolfi-build-essential-release-eng:latest"


def main():
    pipeline_steps = [
        {
            "label": "Validate version bump parameters",
            "key": "validate-version-bump",
            "depends_on": None,
            "agents": {
                "image": WOLFI_IMAGE,
                "cpu": "250m",
                "memory": "512Mi",
            },
            "command": [
                "dev-tools/validate_version_bump_params.sh",
            ],
        },
        {
            "label": "Schedule version bump follow-up steps",
            "key": "schedule-version-bump-follow-up",
            "depends_on": "validate-version-bump",
            "agents": {
                "image": "python",
            },
            "command": [
                "dev-tools/version_bump_upload_phase2.sh",
            ],
        },
    ]

    print(json.dumps({"steps": pipeline_steps}, indent=2))


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
