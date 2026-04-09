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
# It can be integrated into existing or new workflows and includes a plugin
# that polls artifact URLs until the expected version is available.


import contextlib
import json


def main():
    pipeline = {}
    # TODO: replace the block step with version bump logic
    pipeline_steps = [
        {
            "block": "Ready to fetch for DRA artifacts?",
            "prompt": (
                "Unblock when your team is ready to proceed.\n\n"
                "Trigger parameters:\n"
                "- NEW_VERSION: ${NEW_VERSION}\n"
                "- BRANCH: ${BRANCH}\n"
                "- WORKFLOW: ${WORKFLOW}\n"
            ),
            "key": "block-get-dra-artifacts",
            "blocked_state": "running",
        },
        {
            "label": "Fetch DRA Artifacts",
            "key": "fetch-dra-artifacts",
            "depends_on": "block-get-dra-artifacts",
            "agents": {
                "image": "docker.elastic.co/release-eng/wolfi-build-essential-release-eng:latest",
                "cpu": "250m",
                "memory": "512Mi",
                "ephemeralStorage": "1Gi",
            },
            "command": [
                'echo "Starting DRA artifacts retrieval..."',
            ],
            "timeout_in_minutes": 240,
            "retry": {
                "automatic": [
                    {
                        "exit_status": "*",
                        "limit": 2,
                    }
                ],
                "manual": {"permit_on_passed": True},
            },
            "plugins": [
                {
                    "elastic/json-watcher#v1.0.0": {
                        "url": "https://artifacts-staging.elastic.co/ml-cpp/latest/${BRANCH}.json",
                        "field": ".version",
                        "expected_value": "${NEW_VERSION}",
                        "polling_interval": "30",
                    }
                },
                {
                    "elastic/json-watcher#v1.0.0": {
                        "url": "https://storage.googleapis.com/elastic-artifacts-snapshot/ml-cpp/latest/${BRANCH}.json",
                        "field": ".version",
                        "expected_value": "${NEW_VERSION}-SNAPSHOT",
                        "polling_interval": "30",
                    }
                },
            ],
        },
    ]

    pipeline["steps"] = pipeline_steps
    pipeline["notify"] = [
        {
            "slack": {"channels": ["#machine-learn-build"]},
            "if": (
                "(build.branch == 'main' || "
                "build.branch =~ /^[0-9]+\\.[0-9x]+$/) && "
                "(build.state == 'passed' || build.state == 'failed')"
            ),
        },
        {
            "slack": {
                "channels": ["#machine-learn-build"],
                "message": (
                    "Pipeline waiting for approval\n"
                    "Repo: `${REPO}`\n\n"
                    "Ready to fetch DRA artifacts - please unblock when ready.\n"
                    "New version: `${NEW_VERSION}`\n"
                    "Branch: `${BRANCH}`\n"
                    "Workflow: `${WORKFLOW}`\n"
                    "${BUILDKITE_BUILD_URL}\n"
                ),
            },
            "if": 'build.state == "blocked"',
        },
    ]

    print(json.dumps(pipeline, indent=2))


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
