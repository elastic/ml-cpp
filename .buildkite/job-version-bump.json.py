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
# Patch-only: validate NEW_VERSION/BRANCH, verify git push credentials (dry-run),
# open a PR that bumps elasticsearchVersion on BRANCH (see dev-tools/bump_version.sh),
# then poll staging/snapshot artifact JSON until NEW_VERSION appears. The PR must be
# merged (and snapshot/staging builds finish, typically ~1h) while the watcher runs;
# the step allows up to 240 minutes. When DRY_RUN=true the DRA wait step is skipped
# (no change merged, so artifacts would never advance).


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
    # Bump opens a PR; artifacts update after merge + builds. Watcher polls until match or timeout.
    # Skip when DRY_RUN=true (no PR pushed).
    return {
        "label": label,
        "key": key,
        "depends_on": depends_on,
        "if": 'build.env("DRY_RUN") != "true"',
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
            "label": "Verify git push credentials (dry-run)",
            "key": "git-push-auth-probe",
            "depends_on": "validate-version-bump",
            "agents": {
                "image": WOLFI_IMAGE,
                "cpu": "250m",
                "memory": "512Mi",
            },
            "command": [
                "dev-tools/git_push_auth_probe.sh",
            ],
        },
        {
            "label": "Queue a :slack: notification for the pipeline",
            "key": "queue-slack-notify",
            "depends_on": "git-push-auth-probe",
            "command": ".buildkite/pipelines/send_slack_version_bump_notification.sh | buildkite-agent pipeline upload",
            "agents": {
                "image": "python",
            },
        },
        {
            "label": "Bump version to ${NEW_VERSION}",
            "key": "bump-version",
            "depends_on": "queue-slack-notify",
            "agents": {
                "image": WOLFI_IMAGE,
                "cpu": "250m",
                "memory": "512Mi",
            },
            "command": [
                "dev-tools/bump_version.sh",
            ],
        },
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
        ),
    ]

    pipeline = {
        "steps": pipeline_steps,
    }

    print(json.dumps(pipeline, indent=2))


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
