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
# This script generates JSON for the ml-cpp version bump pipeline.
# It is intended to be triggered by the centralized release-eng pipeline.
#
# Patch workflow: bump version on BRANCH, then wait for staging and snapshot
# artifact JSON to publish NEW_VERSION.


import contextlib
import json
import os


WOLFI_IMAGE = "docker.elastic.co/release-eng/wolfi-build-essential-release-eng:latest"
STAGING_URL = "https://artifacts-staging.elastic.co/ml-cpp/latest"
SNAPSHOT_URL = "https://storage.googleapis.com/elastic-artifacts-snapshot/ml-cpp/latest"


def json_watcher_plugin(url, expected_value):
    return {
        "elastic/json-watcher#v1.0.0": {
            "url": url,
            "field": ".version",
            "expected_value": expected_value,
            "polling_interval": "30",
        }
    }


def dra_step(label, key, depends_on, plugins):
    return {
        "label": label,
        "key": key,
        "depends_on": depends_on,
        "agents": {
            "image": WOLFI_IMAGE,
            "cpu": "250m",
            "memory": "512Mi",
            "ephemeralStorage": "1Gi",
        },
        "command": [
            'echo "Waiting for DRA artifacts..."',
        ],
        "timeout_in_minutes": 240,
        "retry": {
            "automatic": [{"exit_status": "*", "limit": 2}],
            "manual": {"permit_on_passed": True},
        },
        "plugins": plugins,
    }


def main():
    pipeline_steps = [
        {
            "label": "Queue a :slack: notification for the pipeline",
            "depends_on": None,
            "command": ".buildkite/pipelines/send_slack_version_bump_notification.sh | buildkite-agent pipeline upload",
            "agents": {
                "image": "python",
            },
        },
        {
            "label": "Bump version to ${NEW_VERSION}",
            "key": "bump-version",
            "agents": {
                "image": WOLFI_IMAGE,
                "cpu": "250m",
                "memory": "512Mi",
            },
            "command": [
                "dev-tools/bump_version.sh",
            ],
        },
    ]

    # Smoke tests: set ML_CPP_VERSION_BUMP_SKIP_DRA_WAIT on the Buildkite build
    # to skip json-watcher polling (avoids a long-running build when NEW_VERSION
    # will never appear in artifact JSON).
    if not os.environ.get("ML_CPP_VERSION_BUMP_SKIP_DRA_WAIT", "").strip():
        pipeline_steps.append(
            dra_step(
                label="Fetch DRA Artifacts",
                key="fetch-dra-artifacts",
                depends_on="bump-version",
                plugins=[
                    json_watcher_plugin(
                        f"{STAGING_URL}/${{BRANCH}}.json",
                        "${NEW_VERSION}",
                    ),
                    json_watcher_plugin(
                        f"{SNAPSHOT_URL}/${{BRANCH}}.json",
                        "${NEW_VERSION}-SNAPSHOT",
                    ),
                ],
            )
        )

    pipeline = {
        "steps": pipeline_steps,
    }

    print(json.dumps(pipeline, indent=2))


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
