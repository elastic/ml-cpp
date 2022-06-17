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

execute_process(COMMAND ${TEST_EXEC} OUTPUT_FILE ${TEST_NAME}.out ERROR_FILE ${TEST_NAME}.out RESULT_VARIABLE TEST_SUCCESS)
if (NOT TEST_SUCCESS EQUAL 0)
  execute_process(COMMAND ${CMAKE_COMMAND} -E cat ${TEST_NAME}.out)
  message(FATAL_ERROR)
endif()
