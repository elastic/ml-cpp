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
# This script generates a pipeline JSON for the ml-cpp pipeline for PR builds.
# Builds for this pipeline may be triggered by commit or comment.
#
# The basic logic of this script is very simple.
# It either parses the github label or the triggering PR comment.
# If a PR comment is present the script ignores the github label.
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

    # Compute which build step keys will exist so that analytics steps
    # can emit a correct depends_on list (not all platforms are built
    # for every PR, depending on labels/comments).
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
        "VERSION_QUALIFIER": "",
        "ML_BUILD_STEP_KEYS": ",".join(build_step_keys),
    }

    if config.build_windows:
        build_windows = pipeline_steps.generate_step_template("Windows", config.action, "", config.build_x86_64)
        pipeline_steps.append(build_windows)
    if config.build_macos:
        build_macos = pipeline_steps.generate_step_template("MacOS", config.action, config.build_aarch64, "")
        pipeline_steps.append(build_macos)
    if config.build_linux:
        build_linux = pipeline_steps.generate_step_template("Linux", config.action, config.build_aarch64, config.build_x86_64)
        pipeline_steps.append(build_linux)

        if config.build_x86_64:
            pipeline_steps.append(pipeline_steps.generate_step("Upload ES tests x86_64 runner pipeline",
                                                               ".buildkite/pipelines/run_es_tests_x86_64.yml.sh"))
            pipeline_steps.append(pipeline_steps.generate_step("Upload ES inference tests x86_64 runner pipeline",
                                                               ".buildkite/pipelines/run_es_inference_tests_x86_64.yml.sh"))
            # We only use linux x86_64 builds for QA tests.
            if config.run_qa_tests:
                pipeline_steps.append(pipeline_steps.generate_step("Upload QA tests runner pipeline",
                                                                   ".buildkite/pipelines/run_qa_tests.yml.sh"))
            if config.run_pytorch_tests:
                pipeline_steps.append(pipeline_steps.generate_step("Upload QA PyTorch tests runner pipeline",
                                                                   ".buildkite/pipelines/run_pytorch_tests.yml.sh"))
            if config.run_serverless_tests:
                pipeline_steps.append(pipeline_steps.generate_step("Upload serverless tests runner pipeline",
                                                                   ".buildkite/pipelines/run_serverless_tests.yml.sh"))
        if config.build_aarch64:
            pipeline_steps.append(pipeline_steps.generate_step("Upload ES tests aarch64 runner pipeline",
                                                               ".buildkite/pipelines/run_es_tests_aarch64.yml.sh"))

    # Check for build timing regressions against nightly baseline
    pipeline_steps.append(pipeline_steps.generate_step("Check build timing regressions",
                                                       ".buildkite/pipelines/check_build_regression.yml.sh",
                                                       soft_fail=True))

    # Validate the PyTorch allowlist against HuggingFace models when
    # triggered from the PyTorch edge pipeline.  Runs in a python:3
    # container since the build/test images don't include Python.
    if config.run_pytorch_tests:
        pipeline_steps.append(pipeline_steps.generate_step("Upload PyTorch allowlist validation",
                                                           ".buildkite/pipelines/validate_pytorch_allowlist.yml.sh",
                                                           soft_fail=True))

    pipeline["env"] = env
    pipeline["steps"] = pipeline_steps
    print(json.dumps(pipeline, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
