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
  execute_process(COMMAND ${TEST_DIR}/${TEST_NAME} $ENV{BOOST_TEST_OUTPUT_FORMAT_FLAGS} --no_color_output  OUTPUT_FILE ${TEST_DIR}/${TEST_NAME}.out ERROR_FILE ${TEST_DIR}/${TEST_NAME}.out RESULT_VARIABLE TEST_SUCCESS)
endif()

if (NOT TEST_SUCCESS EQUAL 0)
  execute_process(COMMAND ${CMAKE_COMMAND} -E cat ${TEST_DIR}/${TEST_NAME}.out)
  file(WRITE "${TEST_DIR}/${TEST_NAME}.failed" "")
endif()
