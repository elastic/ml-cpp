#!/usr/bin/env python3
#
# This script generates a build pipeline for Macos.
#
# OUTPUT:
#   - macos build JSON file.
#

import argparse
import json
import os

from itertools import product

archs = [
  "aarch64",
]
build_types = [
  "RelWithDebInfo",
]
actions = [
  "build",
  "debug"
]


def main(args):
    pipeline_steps = []
    cur_build_types = build_types
    if args.build_type is not None:
        cur_build_types = [args.build_type]

    for arch, build_type in product(archs, cur_build_types):
        pipeline_steps.append({
            "label": f"Build & test :cpp: for MacOS-{arch}-{build_type} :macos:",
            "timeout_in_minutes": "120",
            "agents": {
            },
            "commands": [
              'if [[ "$GITHUB_PR_COMMENT_VAR_ACTION" == "debug" ]]; then export ML_DEBUG=1; fi;',
              f'echo "MacOS {arch} build not yet supported";'
            ],
            "depends_on": "check_style",
            "key": f"build_test_macos-{arch}-{build_type}",
            "env": {
              "CPP_CROSS_COMPILE": "",
              "CMAKE_FLAGS": "-DCMAKE_TOOLCHAIN_FILE=cmake/darwin-arch64.cmake",
              "RUN_TESTS": "true",
              "BOOST_TEST_OUTPUT_FORMAT_FLAGS": "--logger=JUNIT,error,boost_test_results.junit",
            },
            "artifact_paths": "*/*/unittest/boost_test_results.junit",
            #"plugins": {
            #  "test-collector#v1.2.0": {                                                              
            #    "files": "*/*/unittest/boost_test_results.junit",
            #    "format": "junit"
            #  }
            #},
            "notify": [
              {
                "github_commit_status": {
                  "context": f"Build on MacOS {arch} {build_type}",
                },
              },
            ],
        })

    if args.action != "debug":
        pipeline_steps.append({
            "label": "Build :cpp: for macos_x86_64_cross-RelWithDebInfo :macos:",
            "timeout_in_minutes": "120",
            "agents": {
              "cpu": "6",
              "ephemeralStorage": "20G",
              "memory": "64G",
              "image": "docker.elastic.co/ml-dev/ml-macosx-build:16"
            },
            "commands": [
              ".buildkite/scripts/steps/build_and_test.sh"
            ],
            "depends_on": "check_style",
            "key": "build_macos_x86_64_cross-RelWithDebInfo",
            "env": {
              "CPP_CROSS_COMPILE": "macosx",
              "CMAKE_FLAGS": "-DCMAKE_TOOLCHAIN_FILE=cmake/darwin-x86_64.cmake",
              "RUN_TESTS": "false"
            },
            "notify": [
              {
                "github_commit_status": {
                  "context": "Cross compilation build on MacOS x86_64 RelWithDebInfo",
                },
              },
            ],
        })

    pipeline = {
        "steps": pipeline_steps,
    }
    print(json.dumps(pipeline, indent=2))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--build-type',
                        required=False,
                        choices=build_types,
                        default=None,
                        help="Specify a specific build type to build")
    parser.add_argument('--action',
                        required=False,
                        choices=actions,
                        default="build",
                        help="Specify a build action")

    args = parser.parse_args()

    main(args)
