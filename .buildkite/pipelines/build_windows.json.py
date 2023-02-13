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
# This script generates a build pipeline for windows.
#
# OUTPUT:
#   - windows build JSON file.
#

import argparse
import json
import os

from itertools import product

archs = [
    "x86_64",
]
build_types = [
    "RelWithDebInfo",
]
actions = [
    "build",
    "debug"
]
build_snapshot = [
    "true",
    "false"
]

def main(args):
    pipeline_steps = []
    cur_build_types = build_types
    if args.build_type is not None:
        cur_build_types = [args.build_type]

    for arch, build_type in product(archs, cur_build_types):
        pipeline_steps.append({
            "label": f"Build & test :cpp: for Windows-{arch}-{build_type} :windows:",
            "timeout_in_minutes": "120",
            "agents": {
              "provider": "gcp",
              "machineType": "c2-standard-16",
              "minCpuPlatform": "Intel Cascade Lake",
              "image": "family/ml-cpp-1-windows-2016",
            },
            "commands": [
              f'if ( "{args.action}" -eq "debug" ) {{\$Env:ML_DEBUG="1"}}',
              f'if ( "{args.snapshot}" -ne "None" ) {{\\$Env:BUILD_SNAPSHOT="{args.snapshot}"}}',
              f'if ( "{args.version_qualifier}" -ne "None" ) {{\\$Env:VERSION_QUALIFIER="{args.version_qualifier}"}}',
              "Get-ChildItem env:",
              "& .buildkite\\scripts\\steps\\build_and_test.ps1"
            ],
            "depends_on": "check_style",
            "key": f"build_test_Windows-{arch}-{build_type}",
            "env": {
              "CPP_CROSS_COMPILE": "",
              "CMAKE_FLAGS": "-DCMAKE_TOOLCHAIN_FILE=cmake/windows-x86_64.cmake",
              "RUN_TESTS": "true",
              "BOOST_TEST_OUTPUT_FORMAT_FLAGS": "--logger=JUNIT,error,boost_test_results.junit",
            },
            "artifact_paths": ["*/*/unittest/boost_test_results.junit"],
            "plugins": {
              "test-collector#v1.2.0": {
                "files": "*/*/unittest/boost_test_results.junit",
                "format": "junit"
              }
            },
            "notify": [
              {
                "github_commit_status": {
                  "context": f"Build and test on Windows {arch} {build_type}",
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
    parser.add_argument("--snapshot",
                        required=False,
                        choices=build_snapshot,
                        default=None,
                        help="Specify if a snapshot build is wanted.")
    parser.add_argument("--version_qualifier",
                        required=False,
                        default=None,
                        help="Specify a version qualifier.")

    args = parser.parse_args()

    main(args)
