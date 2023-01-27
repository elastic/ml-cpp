#!/usr/bin/env python
#
# This script generates a pipeline JSON for endpoint-dev pipeline.
#
# The basic logic of this script is very simple.
# It either parses the github label or the triggering PR comment.
# If triggered by a PR comment is presented, the script ignores github label.
#

import json

from ml_pipeline import (
    step,
    config as buildConfig,
)


def main():
    pipeline = {}
    pipeline_steps = step.PipelineStep([
        #step.slack_notification,
        step.format_and_validation,
    ])  

    config = buildConfig.Config()
    config.parse()
    if config.build_windows:
        pipeline_steps.append(step.build_windows)
    if config.build_macos:
        pipeline_steps.append(step.build_macos)
    if config.build_linux:
        pipeline_steps.append(step.build_linux)

    pipeline["steps"] = pipeline_steps
    print(json.dumps(pipeline, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
