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
# This script generates a pipeline JSON for the ml-cpp pipeline.
# Builds for this pipeline may be triggered by commit or comment.
#
# The basic logic of this script is very simple.
# It either parses the github label or the triggering PR comment.
# If a PR comment is present the script ignores the github label.
#

import json
import os

from ml_pipeline import (
    step,
    config as buildConfig,
)

env = {
}

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
        build_windows = pipeline_steps.generate_step_template("Windows", config.action, config.snapshot, config.version_qualifier)
        pipeline_steps.append(build_windows)
    if config.build_macos:
        build_macos = pipeline_steps.generate_step_template("MacOS", config.action, config.snapshot, config.version_qualifier)
        pipeline_steps.append(build_macos)
    if config.build_linux:
        build_linux = pipeline_steps.generate_step_template("Linux", config.action, config.snapshot, config.version_qualifier)
        pipeline_steps.append(build_linux)
    # Never run the ES integration tests if VERSION_QUALIFIER has been set.
    if os.environ.get("VERSION_QUALIFIER", "") == "":
        pipeline_steps.append(pipeline_steps.generate_step("Upload ES tests runner pipeline",
                                                           ".buildkite/pipelines/run_es_tests.yml.sh"))
    pipeline_steps.append({"wait": None})
    pipeline_steps.append(pipeline_steps.generate_step("Upload artifact uploader pipeline",
                                                       ".buildkite/pipelines/upload_to_s3.yml.sh"))
    pipeline["env"] = env
    pipeline["steps"] = pipeline_steps
    print(json.dumps(pipeline, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
