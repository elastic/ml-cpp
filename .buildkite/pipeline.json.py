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

from ml_pipeline import (
    step,
    config as buildConfig,
)

env = {
  "BUILD_SNAPSHOT": "yes",
  "VERSION_QUALIFIER": ""
}

wait = {
  "wait": None
}

def main():
    pipeline = {}
    pipeline_steps = step.PipelineStep([
        # TBD
        #step.slack_notification,
        step.format_and_validation,
    ])  

    config = buildConfig.Config()
    config.parse()
    if config.build_windows:
        build_windows = pipeline_steps.generate_step(":windows:", config.action, "windows", config.snapshot, config.candidate)
        pipeline_steps.append(build_windows)
#    if config.build_macos:
#        build_macos = pipeline_steps.generate_step(":macos:", config.action, "macos", config.snapshot, config.candidate)
#        pipeline_steps.append(build_macos)
#    if config.build_linux:
#        build_linux = pipeline_steps.generate_step(":linux:", config.action, "linux", config.snapshot, config.candidate)
#        pipeline_steps.append(build_linux)
#    pipeline_steps.append(step.run_es_tests)
#    pipeline_steps.append(wait)
#    pipeline_steps.append(step.upload_to_s3)

    pipeline["env"] = env
    pipeline["steps"] = pipeline_steps
    print(json.dumps(pipeline, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
