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

execute_process(COMMAND ${CMAKE_COMMAND} -E rm -f ${TEST_DIR}/*.out)
execute_process(COMMAND ${CMAKE_COMMAND} -E rm -f ${TEST_DIR}/*.failed)
execute_process(COMMAND ${CMAKE_COMMAND} -E rm -f boost_test_results*.xml)
execute_process(COMMAND ${CMAKE_COMMAND} -E rm -f boost_test_results*.junit)

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

message(STATUS "SAFE_TEST_NAME=${SAFE_TEST_NAME}")
string(REPLACE "boost_test_results" "boost_test_results${SAFE_TEST_NAME}" BOOST_TEST_OUTPUT_FORMAT_FLAGS "$ENV{BOOST_TEST_OUTPUT_FORMAT_FLAGS}")
set(OUTPUT_FILE "${TEST_DIR}/${TEST_NAME}${SAFE_TEST_NAME}.out")
set(FAILED_FILE "${TEST_DIR}/${TEST_NAME}${SAFE_TEST_NAME}.failed")

# If env var RUN_BOOST_TESTS_IN_FOREGROUND is defined run the tests in the foreground
message(STATUS "RUN_BOOST_TESTS_IN_FOREGROUND=$ENV{RUN_BOOST_TESTS_IN_FOREGROUND}")

if(TEST_NAME STREQUAL "ml_test_seccomp")
  execute_process(COMMAND ${TEST_DIR}/${TEST_NAME} ${TEST_FLAGS} ${TESTS} ${BOOST_TEST_OUTPUT_FORMAT_FLAGS} --logger=HRF,all --report_format=HRF --show_progress=no --no_color_output  OUTPUT_FILE ${OUTPUT_FILE} ERROR_FILE ${OUTPUT_FILE} RESULT_VARIABLE TEST_SUCCESS)
else()
  if(NOT DEFINED ENV{RUN_BOOST_TESTS_IN_FOREGROUND})
    message(STATUS "executing process ${TEST_DIR}/${TEST_NAME} ${TEST_FLAGS} ${TESTS} ${BOOST_TEST_OUTPUT_FORMAT_FLAGS} --no_color_output")
    execute_process(COMMAND ${TEST_DIR}/${TEST_NAME} ${TEST_FLAGS} ${TESTS} ${BOOST_TEST_OUTPUT_FORMAT_FLAGS} --no_color_output  OUTPUT_FILE ${OUTPUT_FILE} ERROR_FILE ${OUTPUT_FILE}  RESULT_VARIABLE TEST_SUCCESS)
  else()
    message(STATUS "executing process ${TEST_DIR}/${TEST_NAME} ${TEST_FLAGS} ${TESTS} ${BOOST_TEST_OUTPUT_FORMAT_FLAGS}")
    execute_process(COMMAND ${TEST_DIR}/${TEST_NAME} ${TEST_FLAGS} ${TESTS} ${BOOST_TEST_OUTPUT_FORMAT_FLAGS}  RESULT_VARIABLE TEST_SUCCESS)
  endif()
endif()

message(STATUS "TESTS EXITED WITH SUCCESS ${TEST_SUCCESS}")

if (NOT TEST_SUCCESS EQUAL 0)
  if (EXISTS ${TEST_DIR}/${TEST_NAME})
    execute_process(COMMAND ${CMAKE_COMMAND} -E cat ${OUTPUT_FILE})
    file(WRITE "${TEST_DIR}/${FAILED_FILE}" "")
  endif()
  message(FATAL_ERROR "Exiting with status ${TEST_SUCCESS}")
endif()

