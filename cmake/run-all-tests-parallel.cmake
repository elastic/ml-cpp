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

# Run ALL test cases from ALL test suites in a single CTest invocation.
#
# This replaces the sequential per-suite approach (test_individually) with a
# unified pool: CTest sees every test case across all suites and schedules
# them optimally across MAX_PROCS parallel slots.
#
# Required variables (passed via -D on command line):
#   SOURCE_DIR  - path to the repository root (for working directories)
#   BUILD_DIR   - path to the CMake build directory
#
# Optional environment variables:
#   MAX_PROCS          - max parallel test processes (default: auto-detect CPUs)
#   MAX_ARGS           - test cases per CTest test (default: 2)
#   BOOST_TEST_OUTPUT_FORMAT_FLAGS - passed through to test executables

cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED SOURCE_DIR OR NOT DEFINED BUILD_DIR)
  message(FATAL_ERROR "SOURCE_DIR and BUILD_DIR must be defined")
endif()

# --- Platform detection ---
if(CMAKE_HOST_WIN32)
  set(_exe_suffix ".exe")
  # Windows needs PATH set for DLL discovery and CPP_SRC_HOME for resource files
  set(ENV{PATH} "${BUILD_DIR}/distribution/platform/windows-x86_64/bin;$ENV{PATH}")
  set(ENV{CPP_SRC_HOME} "${SOURCE_DIR}")
else()
  set(_exe_suffix "")
endif()

# Multi-config generators (Visual Studio, Ninja Multi-Config) place executables
# in a config-specific subdirectory (e.g. RelWithDebInfo/).
if(DEFINED BUILD_TYPE AND NOT BUILD_TYPE STREQUAL "")
  set(_config_subdir "/${BUILD_TYPE}")
else()
  set(_config_subdir "")
endif()

# --- CPU detection ---
cmake_host_system_information(RESULT _num_cpus QUERY NUMBER_OF_LOGICAL_CORES)

if(DEFINED ENV{MAX_PROCS})
  set(_max_procs $ENV{MAX_PROCS})
else()
  # Conservative default: ceil(cpus / 2) for > 4 cores, 2 otherwise
  if(_num_cpus LESS_EQUAL 4)
    set(_max_procs 2)
  else()
    math(EXPR _max_procs "(${_num_cpus} + 1) / 2")
  endif()
endif()

if(DEFINED ENV{MAX_ARGS})
  set(_max_args $ENV{MAX_ARGS})
else()
  set(_max_args 2)
endif()

# Per-test timeout in seconds.  Prevents a single hung or extremely slow test
# batch from consuming the entire step timeout budget and blocking JUnit merge
# and artifact upload.
if(DEFINED ENV{TEST_TIMEOUT})
  set(_test_timeout $ENV{TEST_TIMEOUT})
else()
  set(_test_timeout 2700)
endif()

# --- Discover all test suites ---
# Each test suite is defined by its executable and the source directory
# (which is the working directory for that suite's tests).
set(_suites
  "core:lib/core/unittest"
  "maths_common:lib/maths/common/unittest"
  "maths_time_series:lib/maths/time_series/unittest"
  "maths_analytics:lib/maths/analytics/unittest"
  "model:lib/model/unittest"
  "api:lib/api/unittest"
  "ver:lib/ver/unittest"
  "seccomp:lib/seccomp/unittest"
  "controller:bin/controller/unittest"
  "pytorch_inference:bin/pytorch_inference/unittest"
)

# --- Discover test cases from all suites ---
set(_all_tests "")
set(_test_count 0)

foreach(_suite_entry ${_suites})
  string(REPLACE ":" ";" _parts "${_suite_entry}")
  list(GET _parts 0 _name)
  list(GET _parts 1 _src_dir)

  set(_exe "${BUILD_DIR}/test/${_src_dir}${_config_subdir}/ml_test_${_name}${_exe_suffix}")

  if(NOT EXISTS "${_exe}")
    message(WARNING "Test executable not found: ${_exe}, skipping")
    continue()
  endif()

  # Discover test cases by running --list_content. Temporarily clear
  # BOOST_TEST_OUTPUT_FORMAT_FLAGS so Boost.Test doesn't try to open a
  # JUnit logger during discovery.
  set(_saved_flags "$ENV{BOOST_TEST_OUTPUT_FORMAT_FLAGS}")
  set(ENV{BOOST_TEST_OUTPUT_FORMAT_FLAGS} "")

  # Boost.Test --list_content writes to stderr, so capture both streams
  # into the same variable.
  execute_process(
    COMMAND "${_exe}" --list_content
    WORKING_DIRECTORY "${SOURCE_DIR}/${_src_dir}"
    OUTPUT_VARIABLE _list_out
    ERROR_VARIABLE _list_out
    RESULT_VARIABLE _list_rc
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(ENV{BOOST_TEST_OUTPUT_FORMAT_FLAGS} "${_saved_flags}")

  if(NOT _list_rc EQUAL 0)
    message(WARNING "Failed to list tests from ml_test_${_name}: ${_list_out}")
    continue()
  endif()

  # Parse the Boost.Test --list_content output:
  #   CSuiteTest*
  #       testCase1*
  #       testCase2*
  set(_current_suite "")
  string(REPLACE "\n" ";" _lines "${_list_out}")
  set(_suite_cases "")

  foreach(_line ${_lines})
    # Suite line: starts with C, ends with *
    string(REGEX MATCH "^(C[A-Za-z0-9_]+)\\*" _suite_match "${_line}")
    if(_suite_match)
      string(REGEX REPLACE "\\*$" "" _current_suite "${_suite_match}")
      continue()
    endif()

    # Test case line: indented, ends with *
    string(REGEX MATCH "^[ \t]+([a-zA-Z_][a-zA-Z0-9_]*)\\*" _case_match "${_line}")
    if(_case_match AND _current_suite)
      string(STRIP "${_case_match}" _case_stripped)
      string(REGEX REPLACE "\\*$" "" _case_name "${_case_stripped}")
      list(APPEND _suite_cases "${_current_suite}/${_case_name}")
    endif()
  endforeach()

  list(LENGTH _suite_cases _n_cases)
  message(STATUS "ml_test_${_name}: ${_n_cases} test cases")

  # Store: each test case tagged with its suite name, executable, and working dir
  foreach(_case ${_suite_cases})
    list(APPEND _all_tests "${_name}|${_src_dir}|${_case}")
    math(EXPR _test_count "${_test_count} + 1")
  endforeach()
endforeach()

if(_test_count EQUAL 0)
  message(FATAL_ERROR "No test cases discovered")
endif()

# For multi-config generators (Visual Studio, Ninja Multi-Config), copy test
# executables from the config-specific subdirectory up one level.  Some tests
# (e.g. CProgNameTest::testProgDir) assume the executable resides directly in
# the unittest directory, matching the old ml_add_test() behaviour.
if(NOT _config_subdir STREQUAL "")
  foreach(_suite_entry ${_suites})
    string(REPLACE ":" ";" _parts "${_suite_entry}")
    list(GET _parts 0 _name)
    list(GET _parts 1 _src_dir)
    set(_src_exe "${BUILD_DIR}/test/${_src_dir}${_config_subdir}/ml_test_${_name}${_exe_suffix}")
    set(_dst_exe "${BUILD_DIR}/test/${_src_dir}/ml_test_${_name}${_exe_suffix}")
    if(EXISTS "${_src_exe}")
      file(COPY_FILE "${_src_exe}" "${_dst_exe}")
      message(STATUS "Copied ml_test_${_name} from ${_config_subdir} to parent directory")
    endif()
  endforeach()
endif()

# --- Clean previous results ---
foreach(_suite_entry ${_suites})
  string(REPLACE ":" ";" _parts "${_suite_entry}")
  list(GET _parts 0 _name)
  list(GET _parts 1 _src_dir)

  set(_test_binary_dir "${BUILD_DIR}/test/${_src_dir}")
  file(GLOB _old_out "${_test_binary_dir}/ml_test_${_name}*.out")
  file(GLOB _old_failed "${_test_binary_dir}/ml_test_${_name}*.failed")
  file(GLOB _old_junit "${SOURCE_DIR}/${_src_dir}/boost_test_results*.junit")
  foreach(_f ${_old_out} ${_old_failed} ${_old_junit})
    file(REMOVE "${_f}")
  endforeach()
endforeach()

# --- Group test cases into batches of MAX_ARGS ---
# Batches stay within the same suite (since each suite needs a different
# working directory and executable).
set(_ctest_dir "${BUILD_DIR}/_parallel_tests")
file(MAKE_DIRECTORY "${_ctest_dir}")

set(_ctest_file_content "")
set(_batch_id 0)

foreach(_suite_entry ${_suites})
  string(REPLACE ":" ";" _parts "${_suite_entry}")
  list(GET _parts 0 _name)
  list(GET _parts 1 _src_dir)

  # Collect cases for this suite
  set(_cases_for_suite "")
  foreach(_entry ${_all_tests})
    string(REPLACE "|" ";" _entry_parts "${_entry}")
    list(GET _entry_parts 0 _e_name)
    if(_e_name STREQUAL _name)
      list(GET _entry_parts 2 _e_case)
      list(APPEND _cases_for_suite "${_e_case}")
    endif()
  endforeach()

  list(LENGTH _cases_for_suite _n)
  if(_n EQUAL 0)
    continue()
  endif()

  # Batch them
  set(_batch "")
  set(_batch_count 0)
  foreach(_case ${_cases_for_suite})
    list(APPEND _batch "${_case}")
    math(EXPR _batch_count "${_batch_count} + 1")

    if(_batch_count EQUAL ${_max_args})
      # Emit this batch as a CTest test
      string(REPLACE ";" ":" _run_test "${_batch}")
      math(EXPR _batch_id "${_batch_id} + 1")
      # Sanitize test name for CTest
      list(GET _batch 0 _first_case)
      list(GET _batch -1 _last_case)
      set(_test_label "ml_test_${_name}_${_first_case}:${_last_case}")
      string(REPLACE "/" "_" _test_label "${_test_label}")

      string(APPEND _ctest_file_content
        "add_test(\"${_test_label}\" \"${CMAKE_COMMAND}\""
        " \"-DTEST_DIR=${BUILD_DIR}/test/${_src_dir}\""
        " \"-DTEST_NAME=ml_test_${_name}\""
        " -P \"${SOURCE_DIR}/cmake/test-runner.cmake\")\n"
        "set_tests_properties(\"${_test_label}\" PROPERTIES"
        " WORKING_DIRECTORY \"${SOURCE_DIR}/${_src_dir}\""
        " TIMEOUT ${_test_timeout}"
        " ENVIRONMENT \"TESTS=${_run_test}\")\n"
      )

      set(_batch "")
      set(_batch_count 0)
    endif()
  endforeach()

  # Emit remaining batch
  list(LENGTH _batch _remaining)
  if(_remaining GREATER 0)
    string(REPLACE ";" ":" _run_test "${_batch}")
    math(EXPR _batch_id "${_batch_id} + 1")
    list(GET _batch 0 _first_case)
    list(GET _batch -1 _last_case)
    set(_test_label "ml_test_${_name}_${_first_case}:${_last_case}")
    string(REPLACE "/" "_" _test_label "${_test_label}")

    string(APPEND _ctest_file_content
      "add_test(\"${_test_label}\" \"${CMAKE_COMMAND}\""
      " \"-DTEST_DIR=${BUILD_DIR}/test/${_src_dir}\""
      " \"-DTEST_NAME=ml_test_${_name}\""
      " -P \"${SOURCE_DIR}/cmake/test-runner.cmake\")\n"
      "set_tests_properties(\"${_test_label}\" PROPERTIES"
      " WORKING_DIRECTORY \"${SOURCE_DIR}/${_src_dir}\""
      " TIMEOUT ${_test_timeout}"
      " ENVIRONMENT \"TESTS=${_run_test}\")\n"
    )
  endif()
endforeach()

message(STATUS "Total: ${_test_count} test cases in ${_batch_id} batches across all suites")
message(STATUS "Running with MAX_PROCS=${_max_procs}, MAX_ARGS=${_max_args}, TIMEOUT=${_test_timeout}s (${_num_cpus} logical CPUs)")

# --- Write CTestTestfile.cmake ---
file(WRITE "${_ctest_dir}/CTestTestfile.cmake" "${_ctest_file_content}")

# --- Run CTest ---
execute_process(
  COMMAND ${CMAKE_CTEST_COMMAND}
    --parallel ${_max_procs}
    --output-on-failure
    --test-dir "${_ctest_dir}"
    --no-label-summary
  RESULT_VARIABLE _ctest_rc
)

if(NOT _ctest_rc EQUAL 0)
  message(WARNING "Some tests failed (exit code: ${_ctest_rc})")
endif()

# --- Merge JUnit results per suite ---
# Each batch writes boost_test_results_<batch>.junit in the suite's source
# directory.  Merge them into a single valid boost_test_results.junit per
# suite by extracting <testcase> elements and wrapping in one <testsuite>.
foreach(_suite_entry ${_suites})
  string(REPLACE ":" ";" _parts "${_suite_entry}")
  list(GET _parts 0 _name)
  list(GET _parts 1 _src_dir)

  file(GLOB _junit_files "${SOURCE_DIR}/${_src_dir}/boost_test_results_*.junit")
  list(LENGTH _junit_files _n_junit)
  if(_n_junit EQUAL 0)
    continue()
  endif()

  set(_all_testcases "")
  set(_total_tests 0)
  set(_total_failures 0)
  set(_total_errors 0)
  set(_suite_name "")

  foreach(_f ${_junit_files})
    file(READ "${_f}" _content)

    # Extract suite name from the first file
    if(_suite_name STREQUAL "")
      string(REGEX MATCH "name=\"([^\"]+)\"" _name_match "${_content}")
      if(_name_match)
        string(REGEX REPLACE "name=\"([^\"]+)\"" "\\1" _suite_name "${_name_match}")
      endif()
    endif()

    # Extract all <testcase .../> and <testcase ...>...</testcase> elements,
    # excluding disabled tests (those containing <skipped).
    # Derive failure/error counts from the actual included test cases rather
    # than from the batch-level <testsuite> attributes, to stay consistent
    # with the filtered test count.
    string(REGEX MATCHALL "<testcase [^<]*(/>([\r\n])?|>([^<]|<[^/]|</[^t]|</t[^e])*</testcase>)" _cases "${_content}")
    foreach(_case ${_cases})
      string(FIND "${_case}" "<skipped" _skip_pos)
      if(_skip_pos EQUAL -1)
        string(APPEND _all_testcases "${_case}\n")
        math(EXPR _total_tests "${_total_tests} + 1")
        string(FIND "${_case}" "<failure" _fail_pos)
        if(NOT _fail_pos EQUAL -1)
          math(EXPR _total_failures "${_total_failures} + 1")
        endif()
        string(FIND "${_case}" "<error" _err_pos)
        if(NOT _err_pos EQUAL -1)
          math(EXPR _total_errors "${_total_errors} + 1")
        endif()
      endif()
    endforeach()
  endforeach()

  if(_suite_name STREQUAL "")
    set(_suite_name "ml_test_${_name}")
  endif()

  set(_merged "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
  string(APPEND _merged "<testsuite tests=\"${_total_tests}\" errors=\"${_total_errors}\" failures=\"${_total_failures}\" id=\"0\" name=\"${_suite_name}\">\n")
  string(APPEND _merged "${_all_testcases}")
  string(APPEND _merged "</testsuite>\n")

  file(WRITE "${SOURCE_DIR}/${_src_dir}/boost_test_results.junit" "${_merged}")
  message(STATUS "Merged ${_n_junit} JUnit files for ml_test_${_name}: ${_total_tests} tests, ${_total_failures} failures, ${_total_errors} errors")
endforeach()

# Signal pass/fail for the calling target
if(NOT _ctest_rc EQUAL 0)
  message(FATAL_ERROR "Test failures detected")
endif()
