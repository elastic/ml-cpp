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
# Supports two workflows via the WORKFLOW env var:
#   patch (default) — bump version on BRANCH, wait for 2 artifact sets
#   minor           — create minor branch + bump BRANCH, wait for 3 artifact sets


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
    workflow = os.environ.get("WORKFLOW", "patch")

    pipeline_steps = [
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

    if workflow == "minor":
        # Minor workflow: artifact checks for both the upstream branch and the
        # new minor branch, running in parallel after the bump step.
        #
        # Derive the minor branch from NEW_VERSION: if NEW_VERSION=9.5.0
        # then the previous minor (the new branch) is 9.4 with version 9.4.0.
        new_version = os.environ.get("NEW_VERSION", "0.0.0")
        parts = new_version.split(".")
        if len(parts) >= 2:
            major, minor_num = parts[0], int(parts[1])
            minor_branch = f"{major}.{minor_num - 1}"
            minor_version = f"{major}.{minor_num - 1}.0"
        else:
            minor_branch = "unknown"
            minor_version = "unknown"

        pipeline_steps.append(
            dra_step(
                label=f"Fetch DRA Artifacts (${{BRANCH}})",
                key="fetch-dra-upstream",
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

        pipeline_steps.append(
            dra_step(
                label=f"Fetch DRA Artifacts ({minor_branch})",
                key="fetch-dra-minor",
                depends_on="bump-version",
                plugins=[
                    json_watcher_plugin(
                        f"{STAGING_URL}/{minor_branch}.json",
                        minor_version,
                    ),
                    json_watcher_plugin(
                        f"{SNAPSHOT_URL}/{minor_branch}.json",
                        f"{minor_version}-SNAPSHOT",
                    ),
                ],
            )
        )
    else:
        # Patch workflow: staging + snapshot for BRANCH
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
        "notify": [
            {
                "slack": {"channels": ["#machine-learn-build"]},
                "if": (
                    "(build.branch == 'main' || "
                    "build.branch =~ /^[0-9]+\\.[0-9x]+$/) && "
                    "(build.state == 'passed' || build.state == 'failed')"
                ),
            },
        ],
    }

    print(json.dumps(pipeline, indent=2))


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
