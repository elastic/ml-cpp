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
# This script generates a pipeline JSON for the ml-cpp-branch pipeline for branch builds.
# Builds for this pipeline may be triggered by code, API or UI.
#
import json

from ml_pipeline import (
    step,
    config as buildConfig,
)

def main():
    pipeline = {}
    pipeline_steps = step.PipelineStep([])
    pipeline_steps.append(pipeline_steps.generate_step("Queue a :slack: notification for the pipeline",
                                                       ".buildkite/pipelines/send_slack_notification.sh"))
    pipeline_steps.append(pipeline_steps.generate_step("Queue a :email: notification for the pipeline",
                                                       ".buildkite/pipelines/send_email_notification.sh"))
    pipeline_steps.append(pipeline_steps.generate_step("Upload clang-format validation",
                                                       ".buildkite/pipelines/format_and_validation.yml.sh"))
    config = buildConfig.Config()
    config.parse()
    if config.build_windows:
        build_windows = pipeline_steps.generate_step_template("Windows", "build")
        pipeline_steps.append(build_windows)
    if config.build_macos:
        build_macos = pipeline_steps.generate_step_template("MacOS", "build")
        pipeline_steps.append(build_macos)
    if config.build_linux:
        build_linux = pipeline_steps.generate_step_template("Linux", "build")
        pipeline_steps.append(build_linux)

    pipeline_steps.append({"wait": None})

    # Trigger the DRA pipeline to create and upload artifacts to S3 and GCS
    pipeline_steps.append({"trigger": "ml-cpp-dra",
                           "label": ":rocket: DRA",
                           "async": "true",
                           "build": {
                               "message": "${BUILDKITE_MESSAGE}",
                               "commit": "${BUILDKITE_COMMIT}",
                               "branch": "${BUILDKITE_BRANCH}"}})

    pipeline["steps"] = pipeline_steps
    print(json.dumps(pipeline, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
