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
# Phase 2 of the ml-cpp version-bump pipeline (uploaded by
# dev-tools/version_bump_upload_phase2.sh after validate). Step-level `if` cannot
# use Buildkite meta-data; gating is done in that shell script instead.

import contextlib
import json
import os


WOLFI_IMAGE = "docker.elastic.co/release-eng/wolfi-build-essential-release-eng:latest"


def main():
    pipeline_steps = [
        {
            "label": "Bump version to ${NEW_VERSION}",
            "key": "bump-version",
            "depends_on": "schedule-version-bump-follow-up",
            "agents": {
                "image": WOLFI_IMAGE,
                "cpu": "250m",
                "memory": "512Mi",
            },
            "env": {
                "VERSION_BUMP_MERGE_AUTO": os.environ.get("VERSION_BUMP_MERGE_AUTO", "true"),
            },
            "command": [
                "dev-tools/bump_version.sh",
            ],
        },
        {
            "label": "Notify :slack: — version bump PR needs approval",
            "key": "queue-slack-notify",
            "depends_on": "bump-version",
            "command": [
                ".buildkite/pipelines/send_slack_version_bump_notification.sh",
            ],
            "agents": {
                "image": "python",
            },
        },
        {
            "label": "Fetch DRA Artifacts",
            "key": "fetch-dra-artifacts",
            "depends_on": "queue-slack-notify",
            "agents": {
                "image": WOLFI_IMAGE,
                "cpu": "250m",
                "memory": "512Mi",
                "ephemeralStorage": "1Gi",
            },
            "command": [
                "python3",
                "dev-tools/wait_version_bump_dra.py",
            ],
            "timeout_in_minutes": 240,
            "retry": {
                "automatic": [{"exit_status": "*", "limit": 2}],
                "manual": {"permit_on_passed": True},
            },
        },
    ]

    print(json.dumps({"steps": pipeline_steps}, indent=2))


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
