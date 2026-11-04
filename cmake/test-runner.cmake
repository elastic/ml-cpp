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

# Turn the TEST_FLAGS environment variable into a CMake list variable
if (DEFINED ENV{TEST_FLAGS} AND NOT "$ENV{TEST_FLAGS}" STREQUAL "")
  string(REPLACE " " ";" TEST_FLAGS $ENV{TEST_FLAGS})
endif()

set(SAFE_TEST_NAME "")
set(TESTS "")
# Special case for specifying a subset of tests to run (can be regex)
if (DEFINED ENV{TESTS} AND NOT "$ENV{TESTS}" STREQUAL "")
  set(TESTS "--run_test=$ENV{TESTS}")
  string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" SAFE_TEST_NAME "$ENV{TESTS}")
  set(SAFE_TEST_NAME "_${SAFE_TEST_NAME}")
endif()

string(REPLACE "boost_test_results" "boost_test_results${SAFE_TEST_NAME}" BOOST_TEST_OUTPUT_FORMAT_FLAGS "$ENV{BOOST_TEST_OUTPUT_FORMAT_FLAGS}")
set(OUTPUT_FILE "${TEST_DIR}/${TEST_NAME}${SAFE_TEST_NAME}.out")
set(FAILED_FILE "${TEST_DIR}/${TEST_NAME}${SAFE_TEST_NAME}.failed")

# Clean up only this batch's output files (JUnit cleanup is handled by the
# parent run-all-tests-parallel.cmake before CTest starts).
execute_process(COMMAND ${CMAKE_COMMAND} -E rm -f "${OUTPUT_FILE}")
execute_process(COMMAND ${CMAKE_COMMAND} -E rm -f "${FAILED_FILE}")

# On Unix, run under a watchdog that dumps stacks and kills a hung test just
# before CTest's own timeout would, so the hang can be diagnosed from the log.
set(WATCHDOG_CMD "")
if(DEFINED ENV{ML_TEST_WATCHDOG_SECONDS} AND NOT WIN32)
  find_program(BASH_EXE bash)
  if(BASH_EXE)
    set(WATCHDOG_CMD ${BASH_EXE} ${CMAKE_CURRENT_LIST_DIR}/run-with-watchdog.sh $ENV{ML_TEST_WATCHDOG_SECONDS})
    # macOS SIP strips DYLD_* when bash starts; the wrapper restores it from here
    set(ENV{ML_WATCHDOG_DYLD_LIBRARY_PATH} "$ENV{DYLD_LIBRARY_PATH}")
  endif()
endif()

# If env var RUN_BOOST_TESTS_IN_FOREGROUND is defined run the tests in the foreground
if(TEST_NAME STREQUAL "ml_test_seccomp")
  execute_process(COMMAND ${WATCHDOG_CMD} ${TEST_DIR}/${TEST_NAME} ${TEST_FLAGS} ${TESTS} ${BOOST_TEST_OUTPUT_FORMAT_FLAGS} --logger=HRF,all --report_format=HRF --show_progress=no --no_color_output  OUTPUT_FILE ${OUTPUT_FILE} ERROR_FILE ${OUTPUT_FILE} RESULT_VARIABLE TEST_SUCCESS)
else()
  if(NOT DEFINED ENV{RUN_BOOST_TESTS_IN_FOREGROUND})
    execute_process(COMMAND ${WATCHDOG_CMD} ${TEST_DIR}/${TEST_NAME} ${TEST_FLAGS} ${TESTS} ${BOOST_TEST_OUTPUT_FORMAT_FLAGS} --no_color_output  OUTPUT_FILE ${OUTPUT_FILE} ERROR_FILE ${OUTPUT_FILE}  RESULT_VARIABLE TEST_SUCCESS)
  else()
    execute_process(COMMAND ${TEST_DIR}/${TEST_NAME} ${TEST_FLAGS} ${TESTS} ${BOOST_TEST_OUTPUT_FORMAT_FLAGS}  RESULT_VARIABLE TEST_SUCCESS)
  endif()
endif()

if (NOT TEST_SUCCESS EQUAL 0)
  if (EXISTS ${TEST_DIR}/${TEST_NAME})
    execute_process(COMMAND ${CMAKE_COMMAND} -E cat ${OUTPUT_FILE})
    # Exit code 124 is the watchdog's timeout; run-all-tests-parallel.cmake
    # reads this marker when reporting the batch.
    if(TEST_SUCCESS EQUAL 124)
      file(WRITE "${FAILED_FILE}" "timeout")
    else()
      file(WRITE "${FAILED_FILE}" "")
    endif()
  endif()
  message(FATAL_ERROR "${TEST_NAME}${SAFE_TEST_NAME} failed with exit code ${TEST_SUCCESS}")
endif()

