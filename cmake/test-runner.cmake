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

if(TEST_NAME STREQUAL "ml_test_seccomp")
  execute_process(COMMAND ${TEST_DIR}/${TEST_NAME} $ENV{BOOST_TEST_OUTPUT_FORMAT_FLAGS} --logger=HRF,all --report_format=HRF --show_progress=no --no_color_output  OUTPUT_FILE ${TEST_DIR}/${TEST_NAME}.out ERROR_FILE ${TEST_DIR}/${TEST_NAME}.out RESULT_VARIABLE TEST_SUCCESS)
else()
  # Turn the TEST_FLAGS environment variable into a CMake list variable
  if (DEFINED ENV{TEST_FLAGS} AND NOT "$ENV{TEST_FLAGS}" STREQUAL "")
    string(REPLACE " " ";" TEST_FLAGS $ENV{TEST_FLAGS})
  endif()

  # Special case for specifying a subset of tests to run (can be regex)
  if (DEFINED ENV{TESTS} AND NOT "$ENV{TESTS}" STREQUAL "")
    set(TESTS "--run_test=$ENV{TESTS}")
  endif()

  # If any special command line args are present run the tests in the foreground
  if (DEFINED TEST_FLAGS OR DEFINED TESTS)
    message(STATUS "executing process ${TEST_DIR}/${TEST_NAME} ${TEST_FLAGS} ${TESTS} $ENV{BOOST_TEST_OUTPUT_FORMAT_FLAGS}")
    execute_process(COMMAND ${TEST_DIR}/${TEST_NAME} ${TEST_FLAGS} ${TESTS} $ENV{BOOST_TEST_OUTPUT_FORMAT_FLAGS} RESULT_VARIABLE TEST_SUCCESS)
  else()
    execute_process(COMMAND ${TEST_DIR}/${TEST_NAME} $ENV{TEST_FLAGS} $ENV{BOOST_TEST_OUTPUT_FORMAT_FLAGS}
      --no_color_output  OUTPUT_FILE ${TEST_DIR}/${TEST_NAME}.out ERROR_FILE ${TEST_DIR}/${TEST_NAME}.out RESULT_VARIABLE TEST_SUCCESS)
  endif()
endif()

if (NOT TEST_SUCCESS EQUAL 0)
  execute_process(COMMAND ${CMAKE_COMMAND} -E cat ${TEST_DIR}/${TEST_NAME}.out)
  file(WRITE "${TEST_DIR}/${TEST_NAME}.failed" "")
endif()
