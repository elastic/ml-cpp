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
      "image": os.getenv("DOCKER_IMAGE", "docker.elastic.co/ml-dev/ml-linux-build:34")
   },
   "aarch64": {
      "provider": "aws",
      "instanceType": "m6g.2xlarge",
      "imagePrefix": "core-almalinux-8-aarch64",
      "diskSizeGb": "100",
      "diskName": "/dev/xvda"
   }
}
# Test steps can request less memory since they don't compile
test_agents = {
   "x86_64": {
      "cpu": "6",
      "ephemeralStorage": "20G",
      "memory": "32G",
      "image": os.getenv("DOCKER_IMAGE", "docker.elastic.co/ml-dev/ml-linux-build:34")
   },
   "aarch64": {
      "provider": "aws",
      "instanceType": "m6g.2xlarge",
      "imagePrefix": "core-almalinux-8-aarch64",
      "diskSizeGb": "100",
      "diskName": "/dev/xvda"
   },
}

common_env = {
    "ML_DEBUG": "0",
    "CPP_CROSS_COMPILE": "",
}

def main(args):
    pipeline_steps = []
    cur_build_types = build_types
    if args.build_type is not None:
        cur_build_types = [args.build_type]

    for arch, build_type in product(archs, cur_build_types):
        if args.build_x86_64 and arch == "x86_64" or args.build_aarch64 and arch == "aarch64":

            if arch == "x86_64":
                # x86_64: split into separate build and test steps
                build_key = f"build_linux-{arch}-{build_type}"

                pipeline_steps.append({
                    "label": f"Build :cpp: for linux-{arch}-{build_type} :linux:",
                    "timeout_in_minutes": "180",
                    "agents": agents[arch],
                    "commands": [
                      f'if [[ "{args.action}" == "debug" ]]; then export ML_DEBUG=1; fi',
                      ".buildkite/scripts/steps/build.sh"
                    ],
                    "depends_on": "check_style",
                    "key": build_key,
                    "env": {
                      **common_env,
                      "CMAKE_FLAGS": f"-DCMAKE_TOOLCHAIN_FILE=cmake/linux-{arch}.cmake",
                      "RUN_TESTS": "false",
                    },
                    "notify": [
                      {
                        "github_commit_status": {
                          "context": f"Build on Linux {arch} {build_type}",
                        },
                      },
                    ],
                })

                pipeline_steps.append({
                    "label": f"Test :cpp: for linux-{arch}-{build_type} :linux:",
                    "timeout_in_minutes": "60",
                    "agents": test_agents[arch],
                    "commands": [
                      f'if [[ "{args.action}" == "debug" ]]; then export ML_DEBUG=1; fi',
                      ".buildkite/scripts/steps/run_tests.sh"
                    ],
                    "depends_on": build_key,
                    "key": f"build_test_linux-{arch}-{build_type}",
                    "env": {
                      **common_env,
                      "BUILD_STEP_KEY": build_key,
                      "CMAKE_FLAGS": f"-DCMAKE_TOOLCHAIN_FILE=cmake/linux-{arch}.cmake",
                      "BOOST_TEST_OUTPUT_FORMAT_FLAGS": "--logger=JUNIT,error,boost_test_results.junit",
                    },
                    "plugins": {
                      "test-collector#v1.2.0": {
                        "files": "*/*/unittest/boost_test_results.junit",
                        "format": "junit"
                      }
                    },
                    "notify": [
                      {
                        "github_commit_status": {
                          "context": f"Test on Linux {arch} {build_type}",
                        },
                      },
                    ],
                })
            else:
                # aarch64: split into build and test steps
                build_key = f"build_linux-{arch}-{build_type}"

                pipeline_steps.append({
                    "label": f"Build :cpp: for linux-{arch}-{build_type} :linux:",
                    "timeout_in_minutes": "180",
                    "agents": agents[arch],
                    "commands": [
                      f'if [[ "{args.action}" == "debug" ]]; then export ML_DEBUG=1; fi',
                      ".buildkite/scripts/steps/build.sh"
                    ],
                    "depends_on": "check_style",
                    "key": build_key,
                    "env": {
                      **common_env,
                      "CMAKE_FLAGS": f"-DCMAKE_TOOLCHAIN_FILE=cmake/linux-{arch}.cmake",
                      "RUN_TESTS": "false",
                    },
                    "notify": [
                      {
                        "github_commit_status": {
                          "context": f"Build on Linux {arch} {build_type}",
                        },
                      },
                    ],
                })

                pipeline_steps.append({
                    "label": f"Test :cpp: for linux-{arch}-{build_type} :linux:",
                    "timeout_in_minutes": "60",
                    "agents": test_agents[arch],
                    "commands": [
                      f'if [[ "{args.action}" == "debug" ]]; then export ML_DEBUG=1; fi',
                      ".buildkite/scripts/steps/run_tests.sh"
                    ],
                    "depends_on": build_key,
                    "key": f"build_test_linux-{arch}-{build_type}",
                    "env": {
                      **common_env,
                      "BUILD_STEP_KEY": build_key,
                      "CMAKE_FLAGS": f"-DCMAKE_TOOLCHAIN_FILE=cmake/linux-{arch}.cmake",
                      "BOOST_TEST_OUTPUT_FORMAT_FLAGS": "--logger=JUNIT,error,boost_test_results.junit",
                    },
                    "plugins": {
                      "test-collector#v1.2.0": {
                        "files": "*/*/unittest/boost_test_results.junit",
                        "format": "junit"
                      }
                    },
                    "notify": [
                      {
                        "github_commit_status": {
                          "context": f"Test on Linux {arch} {build_type}",
                        },
                      },
                    ],
                })

    # Add debug build/test steps for PR builds to detect compilation errors with optimization disabled
    if os.environ.get("BUILDKITE_PIPELINE_SLUG", "ml-cpp-pr-builds") != "ml-cpp-debug-build" and \
            os.environ.get("BUILDKITE_PULL_REQUEST", "false") != "false":
        debug_build_key = "build_linux-x86_64-RelWithDebInfo-debug"

        pipeline_steps.append({
            "label": "Build :cpp: for linux-x86_64-RelWithDebInfo (debug) :linux:",
            "timeout_in_minutes": "180",
            "agents": agents["x86_64"],
            "commands": [
              "export ML_DEBUG=1",
              ".buildkite/scripts/steps/build.sh"
            ],
            "depends_on": "check_style",
            "key": debug_build_key,
            "env": {
              **common_env,
              "ML_DEBUG": "1",
              "CMAKE_FLAGS": "-DCMAKE_TOOLCHAIN_FILE=cmake/linux-x86_64.cmake",
              "RUN_TESTS": "false",
              "SKIP_ARTIFACT_UPLOAD": "true",
            },
            "notify": [
              {
                "github_commit_status": {
                  "context": "Build on Linux x86_64 RelWithDebInfo (debug)",
                },
              },
            ],
        })

        pipeline_steps.append({
            "label": "Test :cpp: for linux-x86_64-RelWithDebInfo (debug) :linux:",
            "timeout_in_minutes": "120",
            "agents": test_agents["x86_64"],
            "commands": [
              "export ML_DEBUG=1",
              ".buildkite/scripts/steps/run_tests.sh"
            ],
            "depends_on": debug_build_key,
            "key": "build_test_linux-x86_64-RelWithDebInfo-debug",
            "env": {
              **common_env,
              "BUILD_STEP_KEY": debug_build_key,
              "ML_DEBUG": "1",
              "CMAKE_FLAGS": "-DCMAKE_TOOLCHAIN_FILE=cmake/linux-x86_64.cmake",
              "BOOST_TEST_OUTPUT_FORMAT_FLAGS": "--logger=JUNIT,error,boost_test_results.junit",
            },
            "plugins": {
              "test-collector#v1.2.0": {
                "files": "*/*/unittest/boost_test_results.junit",
                "format": "junit"
              }
            },
            "notify": [
              {
                "github_commit_status": {
                  "context": "Test on Linux x86_64 RelWithDebInfo (debug)",
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
    parser.add_argument("--build-aarch64",
                        required=False,
                        action='store_true',
                        default=False,
                        help="Build for aarch64?.")
    parser.add_argument("--build-x86_64",
                        required=False,
                        action='store_true',
                        default=False,
                        help="Build for x86_64?")
    args = parser.parse_args()

    main(args)
