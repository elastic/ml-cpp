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
# This script generates a pipeline JSON for the ml-cpp-dra pipeline
# which is triggered nightly on a schedule.
# For now the pipeline has just the one step, that combines individual,
# platform specific, zip files into one "uber" zip file.
#

import json

from ml_pipeline import (
    step,
    config as buildConfig,
)

env = {
  "BUILD_SNAPSHOT": "true",
  "VERSION_QUALIFIER": ""
}

def main():
    pipeline = {}
    pipeline["env"] = env

    pipeline_steps = step.PipelineStep([])

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
