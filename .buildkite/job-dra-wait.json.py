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
# Fetch DRA Artifacts step for the ml-cpp patch version-bump workflow. Uploaded
# by dev-tools/maybe_schedule_dra_wait.sh only when bump_version.sh actually
# opened a version-bump PR (ml_cpp_version_bump_changed=true) — a separate
# script because that flag isn't known yet when job-version-bump-phase2.json.py
# itself is generated and uploaded.

import contextlib
import json
import os
import sys


WOLFI_IMAGE = "docker.elastic.co/release-eng/wolfi-build-essential-release-eng:latest"


def main():
    new_version = os.environ.get("NEW_VERSION", "").strip()
    if not new_version:
        print("ERROR: NEW_VERSION must be set", file=sys.stderr)
        sys.exit(1)

    pipeline_steps = [
        {
            "label": "Fetch DRA Artifacts",
            "key": "fetch-dra-artifacts",
            "depends_on": "maybe-schedule-dra-wait",
            "agents": {
                "image": WOLFI_IMAGE,
                "cpu": "250m",
                "memory": "512Mi",
                "ephemeralStorage": "1Gi",
            },
            "plugins": [
                {
                    "elastic/version-bump-dra#v1.0.0": {
                        "product": "ml-cpp",
                        "version": new_version,
                        "workflow": os.environ.get("WORKFLOW", "patch"),
                    }
                }
            ],
            "timeout_in_minutes": 480,
            "retry": {
                "automatic": [{"exit_status": "*", "limit": 2}],
                "manual": {"permit_on_passed": True},
            },
        }
    ]

    print(json.dumps({"steps": pipeline_steps}, indent=2))


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
