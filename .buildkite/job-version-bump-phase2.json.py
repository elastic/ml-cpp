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
import sys


WOLFI_IMAGE = "docker.elastic.co/release-eng/wolfi-build-essential-release-eng:latest"


def main():
    # validate_version_bump_params.sh already requires NEW_VERSION before phase 2
    # is ever uploaded; re-checked here so a broken guarantee fails loudly at
    # generation time instead of baking an empty "version" into the plugin config.
    new_version = os.environ.get("NEW_VERSION", "").strip()
    if not new_version:
        print("ERROR: NEW_VERSION must be set", file=sys.stderr)
        sys.exit(1)

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
                # Same image as bump-version: the minimal python image does not ship
                # buildkite-agent, so meta-data get / pipeline upload silently skipped Slack.
                "image": WOLFI_IMAGE,
                "cpu": "250m",
                "memory": "512Mi",
            },
        },
    ]

    # The plugin derives its target branch from NEW_VERSION alone (9.5.1 -> 9.5)
    # and never sees BRANCH. A sandbox run (BRANCH=testing-9.5, normal-looking
    # NEW_VERSION) would make it poll the real 9.5 branch's production
    # manifests. Skip the step, or risk a false pass or an hours-long wait on a
    # version this run can never publish.
    branch = os.environ.get("BRANCH", "")
    if branch.startswith("testing-"):
        print(
            f"Sandbox release branch {branch!r} — not scheduling the Fetch DRA artifact "
            "step (no real pipeline run will publish artifacts for it).",
            file=sys.stderr,
        )
    else:
        pipeline_steps.append(
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
        )

    print(json.dumps({"steps": pipeline_steps}, indent=2))


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
