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

# run-tests-individually.cmake
#
# Portable replacement for run_tests_as_seperate_processes.sh.
# Discovers Boost.Test cases, generates a temporary CTest project, and
# runs them in parallel via ctest --parallel. Works on Linux, macOS and
# Windows using only CMake built-in functionality (no bash/sed/awk/xargs).
#
# Required -D parameters:
#   BINARY_DIR   - directory containing the test executable
#   TEST_SUITE   - test suite target name without "ml_" prefix, e.g. "test_api"
#   TEST_DIR     - source-side unittest directory (for working directory)
#
# Optional environment variables (same semantics as the shell script):
#   BOOST_TEST_MAX_ARGS   - max test cases per batch (default: 1)
#   BOOST_TEST_MAX_PROCS  - max parallel processes (default: logical CPU count)
#   BOOST_TEST_MIXED_MODE - if "true", batch tests run in one Boost process
#                           using colon-separated --run_test= syntax
#   BOOST_TEST_OUTPUT_FORMAT_FLAGS - passed through to the test executable
#   TEST_FLAGS            - additional flags passed to the test executable

cmake_minimum_required(VERSION 3.19)

# ---------------------------------------------------------------------------
# Validate required parameters
# ---------------------------------------------------------------------------
foreach(_var BINARY_DIR TEST_SUITE TEST_DIR)
  if(NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
    message(FATAL_ERROR "${_var} must be defined. "
      "Usage: cmake -DBINARY_DIR=... -DTEST_SUITE=... -DTEST_DIR=... "
      "-P run-tests-individually.cmake")
  endif()
endforeach()

# ---------------------------------------------------------------------------
# Derive paths
# ---------------------------------------------------------------------------
set(TEST_EXECUTABLE "${BINARY_DIR}/ml_${TEST_SUITE}")
set(LOG_DIR "${BINARY_DIR}/test_logs")
set(CTEST_PROJECT_DIR "${BINARY_DIR}/_ctest_individual")

# ---------------------------------------------------------------------------
# Parallelism and batching settings
# ---------------------------------------------------------------------------
set(MAX_ARGS 2)
if(DEFINED ENV{BOOST_TEST_MAX_ARGS} AND NOT "$ENV{BOOST_TEST_MAX_ARGS}" STREQUAL "")
  set(MAX_ARGS "$ENV{BOOST_TEST_MAX_ARGS}")
endif()

cmake_host_system_information(RESULT _num_cpus QUERY NUMBER_OF_LOGICAL_CORES)
set(MAX_PROCS ${_num_cpus})
if(DEFINED ENV{BOOST_TEST_MAX_PROCS} AND NOT "$ENV{BOOST_TEST_MAX_PROCS}" STREQUAL "")
  set(MAX_PROCS "$ENV{BOOST_TEST_MAX_PROCS}")
endif()

set(MIXED_MODE FALSE)
if(DEFINED ENV{BOOST_TEST_MIXED_MODE} AND "$ENV{BOOST_TEST_MIXED_MODE}" STREQUAL "true")
  set(MIXED_MODE TRUE)
endif()

# Collect extra flags from the environment
set(EXTRA_TEST_FLAGS "")
if(DEFINED ENV{TEST_FLAGS} AND NOT "$ENV{TEST_FLAGS}" STREQUAL "")
  string(REPLACE " " ";" EXTRA_TEST_FLAGS "$ENV{TEST_FLAGS}")
endif()

set(BOOST_OUTPUT_FLAGS "")
if(DEFINED ENV{BOOST_TEST_OUTPUT_FORMAT_FLAGS} AND NOT "$ENV{BOOST_TEST_OUTPUT_FORMAT_FLAGS}" STREQUAL "")
  set(BOOST_OUTPUT_FLAGS "$ENV{BOOST_TEST_OUTPUT_FORMAT_FLAGS}")
endif()

# The seccomp test activates a sandbox that restricts system calls, so we must
# force Human Readable Format (HRF) logging instead of XML/JUNIT which may
# attempt I/O operations the sandbox does not permit.
set(IS_SECCOMP_TEST FALSE)
if(TEST_SUITE STREQUAL "test_seccomp")
  set(IS_SECCOMP_TEST TRUE)
endif()

# ---------------------------------------------------------------------------
# Prepare directories
# ---------------------------------------------------------------------------
file(REMOVE_RECURSE "${LOG_DIR}")
file(MAKE_DIRECTORY "${LOG_DIR}")
file(REMOVE_RECURSE "${CTEST_PROJECT_DIR}")
file(MAKE_DIRECTORY "${CTEST_PROJECT_DIR}")

# ---------------------------------------------------------------------------
# Discover tests via --list_content
# ---------------------------------------------------------------------------
message(STATUS "Discovering tests from ${TEST_EXECUTABLE}...")
execute_process(
  COMMAND "${TEST_EXECUTABLE}" --list_content
  OUTPUT_VARIABLE _list_output
  ERROR_VARIABLE  _list_output
  RESULT_VARIABLE _list_result
)

if(NOT _list_result EQUAL 0 AND "${_list_output}" STREQUAL "")
  message(FATAL_ERROR "Failed to discover tests from ${TEST_EXECUTABLE}")
endif()

# Parse Suite/Case names from --list_content output.
# Boost.Test --list_content produces output like:
#   CSomeTest*
#       testSomething*
#       testAnotherThing*
#   CAnotherTest*
#       testFoo*
set(ALL_TEST_NAMES "")
set(_current_suite "")
string(REPLACE "\n" ";" _lines "${_list_output}")
foreach(_line IN LISTS _lines)
  string(STRIP "${_line}" _stripped)
  if(_stripped MATCHES "^(C.*Test)\\*$")
    set(_current_suite "${CMAKE_MATCH_1}")
  elseif(_stripped MATCHES "^(test.*)\\*$" AND NOT "${_current_suite}" STREQUAL "")
    list(APPEND ALL_TEST_NAMES "${_current_suite}/${CMAKE_MATCH_1}")
  endif()
endforeach()

list(LENGTH ALL_TEST_NAMES _num_tests)
if(_num_tests EQUAL 0)
  message(FATAL_ERROR "No tests found to run or error in test discovery.")
endif()
message(STATUS "Discovered ${_num_tests} test(s)")

# ---------------------------------------------------------------------------
# Group tests into batches of MAX_ARGS
#
# We use a pipe "|" delimiter within each batch because semicolons are
# CMake list separators and colons are Boost.Test --run_test separators.
# ---------------------------------------------------------------------------
set(_batches "")
set(_batch_idx 0)
set(_count 0)
set(_current_batch "")

foreach(_test IN LISTS ALL_TEST_NAMES)
  if(_count GREATER_EQUAL ${MAX_ARGS})
    list(APPEND _batches "${_current_batch}")
    set(_current_batch "")
    set(_count 0)
    math(EXPR _batch_idx "${_batch_idx} + 1")
  endif()
  if("${_current_batch}" STREQUAL "")
    set(_current_batch "${_test}")
  else()
    set(_current_batch "${_current_batch}|${_test}")
  endif()
  math(EXPR _count "${_count} + 1")
endforeach()
if(NOT "${_current_batch}" STREQUAL "")
  list(APPEND _batches "${_current_batch}")
endif()

list(LENGTH _batches _num_batches)
message(STATUS "Running ${_num_tests} test(s) in ${_num_batches} batch(es), "
  "max ${MAX_PROCS} parallel process(es)")

# ---------------------------------------------------------------------------
# Generate a per-batch runner script invoked by each CTest test entry
# ---------------------------------------------------------------------------
set(_batch_runner "${CTEST_PROJECT_DIR}/_run_one_batch.cmake")
file(WRITE "${_batch_runner}" [=[
# Per-batch runner script.
# Invoked by CTest with -D parameters for each batch.
cmake_minimum_required(VERSION 3.19)

# Build the command line
set(_cmd "${TEST_EXECUTABLE}" "--run_test=${RUN_TEST_ARG}" --no_color_output)

# The seccomp test activates a sandbox that restricts system calls.
# Force HRF logging to avoid I/O operations the sandbox does not permit.
if(IS_SECCOMP_TEST)
  list(APPEND _cmd --logger=HRF,all --report_format=HRF --show_progress=no)
endif()

# Append extra test flags if provided
if(NOT "${EXTRA_TEST_FLAGS}" STREQUAL "")
  string(REPLACE ";" " " _flags_str "${EXTRA_TEST_FLAGS}")
  string(REPLACE " " ";" _flags_list "${_flags_str}")
  list(APPEND _cmd ${_flags_list})
endif()

# Append Boost output format flags if provided (skip for seccomp as it uses HRF)
if(NOT "${BOOST_OUTPUT_FLAGS}" STREQUAL "" AND NOT IS_SECCOMP_TEST)
  # Substitute the test name into the output format flags so each
  # batch writes to its own results file
  string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" _safe_name "${RUN_TEST_ARG}")
  string(REPLACE "boost_test_results" "boost_test_results_${_safe_name}" _output_flags "${BOOST_OUTPUT_FLAGS}")
  string(REPLACE " " ";" _output_flags_list "${_output_flags}")
  list(APPEND _cmd ${_output_flags_list})
endif()

execute_process(
  COMMAND ${_cmd}
  OUTPUT_FILE "${LOG_FILE}"
  ERROR_FILE  "${LOG_FILE}"
  RESULT_VARIABLE _result
  WORKING_DIRECTORY "${WORKING_DIR}"
)

if(NOT _result EQUAL 0)
  file(READ "${LOG_FILE}" _log_content)
  message("${_log_content}")
  message(FATAL_ERROR "Test(s) '${RUN_TEST_ARG}' FAILED with exit code ${_result}")
endif()
]=])

# ---------------------------------------------------------------------------
# Generate CTestTestfile.cmake with one add_test() per batch
# ---------------------------------------------------------------------------
set(_ctest_file "${CTEST_PROJECT_DIR}/CTestTestfile.cmake")
file(WRITE "${_ctest_file}" "# Auto-generated by run-tests-individually.cmake\n\n")

set(_idx 0)
foreach(_batch IN LISTS _batches)
  # In mixed mode or multi-test batches, join with ":" for Boost.Test
  if(MIXED_MODE)
    string(REPLACE "|" ":" _run_test_arg "${_batch}")
  else()
    # With MAX_ARGS=1, _batch is just a single test name.
    # With MAX_ARGS>1 and not mixed mode, each test in the batch
    # still needs to run individually. However, CTest gives us
    # per-entry parallelism, so for simplicity (matching the shell
    # script behaviour) we join with ":" and run in one Boost process.
    string(REPLACE "|" ":" _run_test_arg "${_batch}")
  endif()

  # Safe log filename
  string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" _safe_name "${_run_test_arg}")
  string(SUBSTRING "${_safe_name}" 0 100 _safe_name)
  set(_log_file "${LOG_DIR}/${_safe_name}.log")

  # Use the test name as the CTest test name for readable output
  set(_test_label "${_run_test_arg}")

  # Escape semicolons in EXTRA_TEST_FLAGS for -D passing
  string(REPLACE ";" "\\;" _escaped_flags "${EXTRA_TEST_FLAGS}")

  file(APPEND "${_ctest_file}"
    "add_test(\"${_test_label}\" \"${CMAKE_COMMAND}\""
    " \"-DRUN_TEST_ARG=${_run_test_arg}\""
    " \"-DTEST_EXECUTABLE=${TEST_EXECUTABLE}\""
    " \"-DLOG_FILE=${_log_file}\""
    " \"-DWORKING_DIR=${TEST_DIR}\""
    " \"-DEXTRA_TEST_FLAGS=${_escaped_flags}\""
    " \"-DBOOST_OUTPUT_FLAGS=${BOOST_OUTPUT_FLAGS}\""
    " \"-DIS_SECCOMP_TEST=${IS_SECCOMP_TEST}\""
    " -P \"${_batch_runner}\")\n"
    "set_tests_properties(\"${_test_label}\" PROPERTIES WORKING_DIRECTORY \"${TEST_DIR}\")\n\n"
  )

  math(EXPR _idx "${_idx} + 1")
endforeach()

message(STATUS "Generated CTest project with ${_num_batches} test(s)")
message(STATUS "Running with ctest --parallel ${MAX_PROCS}...")
message(STATUS "--------------------------------------------------")

# ---------------------------------------------------------------------------
# Run ctest --parallel for true concurrent execution
# ---------------------------------------------------------------------------
execute_process(
  COMMAND "${CMAKE_CTEST_COMMAND}"
    --test-dir "${CTEST_PROJECT_DIR}"
    --parallel ${MAX_PROCS}
    --output-on-failure
    --no-label-summary
    --progress
  RESULT_VARIABLE _ctest_result
  WORKING_DIRECTORY "${TEST_DIR}"
)

message(STATUS "--------------------------------------------------")

if(NOT _ctest_result EQUAL 0)
  message(STATUS "${TEST_SUITE}: Some individual tests FAILED. Check logs in '${LOG_DIR}'.")
else()
  message(STATUS "${TEST_SUITE}: All individual tests PASSED.")
endif()

# ---------------------------------------------------------------------------
# Clean up temporary CTest project
# ---------------------------------------------------------------------------
file(REMOVE_RECURSE "${CTEST_PROJECT_DIR}")

# ---------------------------------------------------------------------------
# Merge JUnit results if requested
# ---------------------------------------------------------------------------
if(NOT "${BOOST_OUTPUT_FLAGS}" STREQUAL "")
  string(FIND "${BOOST_OUTPUT_FLAGS}" "junit" _junit_pos)
  if(NOT _junit_pos EQUAL -1)
    file(GLOB _junit_files "${TEST_DIR}/boost_test_results_C*.junit")
    list(LENGTH _junit_files _num_junit)
    if(_num_junit GREATER 0)
      message(STATUS "Merging ${_num_junit} JUnit result file(s)...")

      set(_total_tests 0)
      set(_total_errors 0)
      set(_total_failures 0)
      set(_suite_name "")
      set(_suite_id "")
      set(_all_testcases "")

      foreach(_jf IN LISTS _junit_files)
        file(READ "${_jf}" _jc)

        if(_jc MATCHES "tests=\"([0-9]+)\"")
          math(EXPR _total_tests "${_total_tests} + ${CMAKE_MATCH_1}")
        endif()
        if(_jc MATCHES "errors=\"([0-9]+)\"")
          math(EXPR _total_errors "${_total_errors} + ${CMAKE_MATCH_1}")
        endif()
        if(_jc MATCHES "failures=\"([0-9]+)\"")
          math(EXPR _total_failures "${_total_failures} + ${CMAKE_MATCH_1}")
        endif()
        if("${_suite_name}" STREQUAL "" AND _jc MATCHES "name=\"([a-zA-Z.]+)\"")
          set(_suite_name "${CMAKE_MATCH_1}")
        endif()
        if("${_suite_id}" STREQUAL "" AND _jc MATCHES "id=\"([0-9]+)\"")
          set(_suite_id "${CMAKE_MATCH_1}")
        endif()

        # Extract non-skipped testcase elements
        string(REGEX MATCHALL "<testcase[^/]*(/|[^>]*</testcase)>" _cases "${_jc}")
        foreach(_case IN LISTS _cases)
          string(FIND "${_case}" "skipped" _skip_pos)
          if(_skip_pos EQUAL -1)
            string(APPEND _all_testcases "${_case}\n")
          endif()
        endforeach()
      endforeach()

      set(_merged_file "${TEST_DIR}/boost_test_results.junit")
      file(WRITE "${_merged_file}"
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<testsuite tests=\"${_total_tests}\" skipped=\"0\""
        " errors=\"${_total_errors}\" failures=\"${_total_failures}\""
        " id=\"${_suite_id}\" name=\"${_suite_name}\">\n"
        "${_all_testcases}"
        "</testsuite>\n"
      )
      message(STATUS "Merged JUnit results written to ${_merged_file}")
    endif()
  endif()
endif()

# Propagate failure to the calling build system
if(NOT _ctest_result EQUAL 0)
  message(FATAL_ERROR "Test failures detected")
endif()
