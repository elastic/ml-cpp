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
    pipeline["env"] = env

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

    # Build the DRA artifacts and upload to S3 and GCS
    pipeline_steps.append(pipeline_steps.generate_step("Create daily releasable artifacts",
                                                       ".buildkite/pipelines/create_dra.yml.sh"))
    pipeline_steps.append(pipeline_steps.generate_step("Upload daily releasable artifacts to S3",
                                                       ".buildkite/pipelines/upload_dra_to_s3.yml.sh"))
    pipeline_steps.append(pipeline_steps.generate_step("Upload daily releasable artifacts to GCS",
                                                       ".buildkite/pipelines/upload_dra_to_gcs.yml.sh"))

    pipeline["steps"] = pipeline_steps
    print(json.dumps(pipeline, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
