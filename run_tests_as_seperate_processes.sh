#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#

# This script ultimately gets called from within the docker entry point script.
# It provides a wrapper around the call to "cmake" that runs the test cases
# and provides some flexibility as to how the tests should be run in terms of how they
# are spread across processes. This is necessary when trying to isolate the impact memory
# usage of tests have upon one another.
#
# It is intended to be called as part of the CI build/test process but should be able to be run manually.
#
# It should be called with 3 parameters
# cmake_build_dir: The directory that cmake is using for build outputs, i.e. that passed to cmake's --build argument
# cmake_current_binary_dir: The directory containing the current test suite executable e.g. <cmake_build_dir>/test/lib/api/unittest
# test_suite: The name of the test suite to run, minus any leading "ml_", e.g. "test_api"
#
# In addition to the required parameters there are several environment variables that control the script's behaviour
# BOOST_TEST_MAX_ARGS: The maximum number of test cases to be passed off to a sub shell
# BOOST_TEST_MAX_PROCS: The maximum number of sub shells to use
# BOOST_TEST_MIXED_MODE: If set to "true" then rather than iterating over each individual test passed to a sub-shell
#                        run them all in the same BOOST test executable process.
#
# Design decisions: The script relies upon the simplest tools available on most unix like platforms - bash, sed and
# awk (the awk script does not use any GNU extensions for maximum portability). This is to keep the number of dependencies
# required by CI build images to a minimum (so e.g. no python etc.)

if [ $# -lt 3 ]; then
  echo "Usage: $0 <cmake_build_dir> <cmake_current_binary_dir> <test_suite>"
  echo "e.g.: $0 ${CPP_SRC_HOME}/cmake-build-relwithdebinfo-local ${CPP_SRC_HOME}/cmake-build-relwithdebinfo-local/test/lib/api/unittest test_api"
  exit
fi

export BUILD_DIR=$1
export BINARY_DIR=$2
export TEST_SUITE=$3

TEST_DIR=${CPP_SRC_HOME}/$(echo $BINARY_DIR | sed "s|$BUILD_DIR/test/||")

export TEST_EXECUTABLE="$2/ml_$3"
export LOG_DIR="$2/test_logs"

MAX_ARGS=2
MAX_PROCS=4

if [[ -n "$BOOST_TEST_MAX_ARGS" ]]; then
  MAX_ARGS=$BOOST_TEST_MAX_ARGS
fi

if [[ -n "$BOOST_TEST_MAX_PROCS" ]]; then
  MAX_PROCS=$BOOST_TEST_MAX_PROCS
fi

rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

function get_qualified_test_names() {
    executable_path=$1

    output_lines=$($executable_path --list_content 2>&1)

    while IFS= read -r line; do
      match=$(grep -w '^[ ]*C.*Test' <<< "$line");
      if [ $? -eq 0 ]; then
        suite=$match
        continue
      fi
      match=$(grep -w 'test.*\*$' <<< "$line");
      if [ $? -eq 0 ]; then
        case=$(sed 's/[ \*]//g' <<< "$suite/$match")
        echo "$case"
      fi
    done <<< "$output_lines"
}

# get the fully qualified test names
echo "Discovering tests..."
ALL_TEST_NAMES=$(get_qualified_test_names "$TEST_EXECUTABLE")

if [ -z "$ALL_TEST_NAMES" ]; then
    echo "No tests found to run or error in test discovery."
    exit 1
fi

EXIT_CODE=0
export RUN_BOOST_TESTS_IN_BACKGROUND=1

function execute_tests() {

  if [[ "$BOOST_TEST_MIXED_MODE" == "true" ]]; then
    TEST_CASES=$(sed 's/ /:/g' <<< $@)
  else
    TEST_CASES=$@
  fi

  # Loop through each test
  for TEST_NAME in $TEST_CASES; do
        echo "--------------------------------------------------"
        echo "Running test: $TEST_NAME"

        # Replace slashes and potentially other special chars for a safe filename
        SAFE_TEST_LOG_FILENAME=$(echo "$TEST_NAME" | sed 's/[^a-zA-Z0-9_]/_/g' | cut -c-100)
        LOG_FILE="$LOG_DIR/${SAFE_TEST_LOG_FILENAME}.log"

        # Execute the test in a separate process
        TESTS=$TEST_NAME cmake --build $BUILD_DIR -t $TEST_SUITE > "$LOG_FILE" 2>&1
        TEST_STATUS=$?

        if [ $TEST_STATUS -eq 0 ]; then
            echo "Test '$TEST_NAME' PASSED."
        else
            echo "Test '$TEST_NAME' FAILED with exit code $TEST_STATUS. Check '$LOG_FILE' for details."
            EXIT_CODE=1 # Indicate overall failure if any test fails
        fi
    done
}

export -f execute_tests

echo $ALL_TEST_NAMES | xargs -n $MAX_ARGS -P $MAX_PROCS bash -c 'execute_tests "$@"' _

echo "--------------------------------------------------"
if [ $EXIT_CODE -eq 0 ]; then
    echo "$TEST_SUITE: All individual tests PASSED."
else
    echo "$TEST_SUITE: Some individual tests FAILED. Check logs in '$LOG_DIR'."
fi

function merge_junit_results() {
  JUNIT_FILES="$@"
  echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
  cat $JUNIT_FILES | \
    awk '
    BEGIN{tests=0; skipped=0; errors=0; failures=0; id=""; time=0.0; name=""}
    $0 ~ /<testsuite.*/ {
      match($2, "[0-9]+")
      { tests+=substr($2, RSTART, RLENGTH) }
      match($3, "[0-9]+")
      { skipped=substr($3, RSTART, RLENGTH) }
      match($4, "[0-9]+")
      { errors+=substr($4, RSTART, RLENGTH) }
      match($5, "[0-9]+")
      { failures+=substr($5, RSTART, RLENGTH) }
      match($6, "[0-9]+")
      { id=substr($6, RSTART, RLENGTH) }
      match($7, "\"[a-zA-Z.]+\"")
      { name=substr($7, RSTART+1, RLENGTH-2) }
      match($8, "[0-9]+.[0-9]+")
      { time+=substr($8, RSTART, RLENGTH) }
    }
    END{print "<testsuite", "tests=\""tests"\"", "skipped=\"0\"", "errors=\""errors"\"", "failures=\""failures"\"", "id=\""id"\"", "name=\""name"\"", "time=\""time"\"" ">"}'

  cat $JUNIT_FILES | sed -e '/xml/d' -e '/testsuite/d' -e '/<testcase/i\
' -e '/<\/testcase/a\
' | sed -e '/<testcase/,/testcase>/{H;d;};x;/skipped/d' | grep '.'
echo "</testsuite>"
echo
}

merge_junit_results $TEST_DIR/boost_test_results_C*.junit > $TEST_DIR/boost_test_results.junit

exit $EXIT_CODE
