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
# This script generates a pipeline JSON for the ml-cpp-debug-build pipeline
# which is triggered nightly on a schedule. The build itself runs with
# full debug and assertions enabled.
#
# The basic logic of this script is very simple.
# It either parses the github label or the triggering PR comment.
# If a PR comment is present, the script ignores the github label.
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

    build_step_keys = []
    if config.build_linux and config.build_aarch64:
        build_step_keys.append("build_test_linux-aarch64-RelWithDebInfo")
    if config.build_linux and config.build_x86_64:
        build_step_keys.append("build_test_linux-x86_64-RelWithDebInfo")
    if config.build_macos and config.build_aarch64:
        build_step_keys.append("build_test_macos-aarch64-RelWithDebInfo")
    if config.build_windows and config.build_x86_64:
        build_step_keys.append("build_test_Windows-x86_64-RelWithDebInfo")

    env = {
        "BUILD_SNAPSHOT": "true",
        "VERSION_QUALIFIER": "",
        "ML_BUILD_STEP_KEYS": ",".join(build_step_keys),
    }

    if config.build_windows:
        debug_windows = pipeline_steps.generate_step_template("Windows", "debug", "", config.build_x86_64)
        pipeline_steps.append(debug_windows)
    if config.build_macos:
        debug_macos = pipeline_steps.generate_step_template("MacOS", "debug", config.build_aarch64, "")
        pipeline_steps.append(debug_macos)
    if config.build_linux:
        debug_linux = pipeline_steps.generate_step_template("Linux", "debug", config.build_aarch64, config.build_x86_64)
        pipeline_steps.append(debug_linux)

    if config.run_pytorch_tests:
        pipeline_steps.append(pipeline_steps.generate_step("Upload PyTorch tests runner pipeline",
                                                           ".buildkite/pipelines/run_pytorch_tests.yml.sh"))

    # Ingest step-level timings into Elasticsearch for anomaly detection
    pipeline_steps.append(pipeline_steps.generate_step("Ingest build timings",
                                                       ".buildkite/pipelines/ingest_build_timings.yml.sh"))
    # Analyze failures with AI if the build failed
    pipeline_steps.append(pipeline_steps.generate_step("Analyze build failure",
                                                       ".buildkite/pipelines/analyze_build_failure.yml.sh"))

    pipeline["env"] = env
    pipeline["steps"] = pipeline_steps
    print(json.dumps(pipeline, indent=2))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
