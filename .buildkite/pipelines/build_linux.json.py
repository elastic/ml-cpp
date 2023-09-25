#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
# This script generates a build pipeline for linux.
#
# OUTPUT:
#   - linux build JSON file.
#

import argparse
import json
import os

from itertools import product

archs = [
    "x86_64",
    "aarch64",
]
build_types = [
    "RelWithDebInfo",
]
actions = [
    "build",
    "debug"
]
agents = {
   "x86_64": {
      "cpu": "6",
      "ephemeralStorage": "20G",
      "memory": "64G",
      "image": "docker.elastic.co/ml-dev/ml-linux-build:27"
   },
   "aarch64": {
      "provider": "aws",
      "instanceType": "m6g.2xlarge",
      "imagePrefix": "ci-amazonlinux-2-aarch64",
      "diskSizeGb": "100",
      "diskName": "/dev/xvda"
   }
}

def main(args):
    pipeline_steps = []
    cur_build_types = build_types
    if args.build_type is not None:
        cur_build_types = [args.build_type]

    for arch, build_type in product(archs, cur_build_types):
        pipeline_steps.append({
            "label": f"Build & test :cpp: for linux-{arch}-{build_type} :linux:",
            "timeout_in_minutes": "240",
            "agents": agents[arch],
            "commands": [
              f'if [[ "{args.action}" == "debug" ]]; then export ML_DEBUG=1; fi',
              ".buildkite/scripts/steps/build_and_test.sh"
            ],
            "depends_on": "check_style",
            "key": f"build_test_linux-{arch}-{build_type}",
            "env": {
              "ML_DEBUG": "0",
              "CMAKE_FLAGS": f"-DCMAKE_TOOLCHAIN_FILE=cmake/linux-{arch}.cmake",
              "CPP_CROSS_COMPILE": "",
              "RUN_TESTS": "true",
              "BOOST_TEST_OUTPUT_FORMAT_FLAGS": "--logger=JUNIT,error,boost_test_results.junit",
            },
            "artifact_paths": "*/**/unittest/boost_test_results.junit",
            "plugins": {
              "test-collector#v1.2.0": {                                                              
                "files": "*/*/unittest/boost_test_results.junit",
                "format": "junit"
              }
            },
            "notify": [
              {
                "github_commit_status": {
                  "context": f"Build and test on Linux {arch} {build_type}",
                },
              },
            ],
        })

    # Never cross-compile for linux-aarch64 in the nightly debug build.
    if os.environ.get("BUILDKITE_PIPELINE_SLUG", "ml-cpp-pr-builds") != "ml-cpp-debug-build" and \
            os.environ.get("BUILDKITE_PULL_REQUEST", "false") != "false":
        # Always cross compile for aarch64 with full debug and assertions
        # enabled for PR builds only. This is to detect any compilation errors
        # as early as possible.
        pipeline_steps.append({
            "label": "Build :cpp: for linux_aarch64_cross-RelWithDebInfo :linux:",
            "timeout_in_minutes": "240",
            "agents": {
              "cpu": "6",
              "ephemeralStorage": "20G",
              "memory": "64G",
              "image": "docker.elastic.co/ml-dev/ml-linux-aarch64-cross-build:11"
            },
            "commands": [
              ".buildkite/scripts/steps/build_and_test.sh"
            ],
            "depends_on": "check_style",
            "key": "build_linux_aarch64_cross-RelWithDebInfo",
            "env": {
              "CPP_CROSS_COMPILE": "aarch64",
              "CMAKE_FLAGS": "-DCMAKE_TOOLCHAIN_FILE=cmake/linux-aarch64.cmake",
              "RUN_TESTS": "false",
              "ML_DEBUG": "1"
            },
            "notify": [
              {
                "github_commit_status": {
                  "context": "Cross compile for Linux aarch64 RelWithDebInfo",
                },
              },
            ],
        })

    pipeline = {
        "steps": pipeline_steps,
    }
    print(json.dumps(pipeline, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-type",
                        required=False,
                        choices=build_types,
                        default=None,
                        help="Specify a specific build type to build")
    parser.add_argument("--action",
                        required=False,
                        choices=actions,
                        default="build",
                        help="Specify a build action.")

    args = parser.parse_args()

    main(args)
