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

file(GLOB_RECURSE TEST_RESULTS "${TEST_DIR}/*.failed" )
if (TEST_RESULTS)
  foreach(TEST_RESULT ${TEST_RESULTS})
    if(${TEST_RESULT} MATCHES "/([^/]+).failed")
      list(APPEND FAILED_TESTS ${CMAKE_MATCH_1})
    endif()
  endforeach()
  list(JOIN FAILED_TESTS ", " FAILURE_STRING)
  message(FATAL_ERROR "Test failures: ${FAILURE_STRING}")
endif()
