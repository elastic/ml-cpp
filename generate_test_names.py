#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# This script provides a wrapper around a call to a BOOST test executable
# to return a formatted list of tests such that each fully qualified test
# name would be in a form suitable to being passed to BOOST test's "--run_test"
# parameter.
# It takes precisely one positional parameter, the path to a BOOST test executable.


import argparse
import re
import subprocess
import sys


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('exec_path', help='The path to the ml_test suite executable')
    return parser.parse_args()

def get_qualified_test_names(executable_path):

    cmd = [args.exec_path, "--list_content"]
    process = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output_lines = process.stderr.splitlines()

    test_names = []
    current_suite_stack = []

    for line in output_lines:
        match_suite = re.match(r'^( *)(C.*Test)\*$', line)
        match_case = re.match(r'^( *)(test.*)\*$', line)

        if match_suite:
            indent_level = len(match_suite.group(1))
            suite_name = match_suite.group(2)

            # Pop suites from stack if current indent is less or equal
            while current_suite_stack and len(current_suite_stack[-1][0]) >= indent_level:
                current_suite_stack.pop()

            current_suite_stack.append((match_suite.group(1), suite_name))
        elif match_case:
            indent_level = len(match_case.group(1))
            case_name = match_case.group(2)

            # Pop suites from stack if current indent is less (for sibling suites/cases)
            while current_suite_stack and len(current_suite_stack[-1][0]) >= indent_level:
                current_suite_stack.pop()

            full_path = "/".join([s[1] for s in current_suite_stack] + [case_name])
            test_names.append(full_path)
    return test_names

if __name__ == "__main__":
    args = parse_arguments()
    try:
        names = get_qualified_test_names(args.exec_path)
        for name in names:
            print(name)
    except subprocess.CalledProcessError as e:
        print(f"Error listing tests: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Test executable '{args.exec_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
