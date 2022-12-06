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

ml_add_test(lib/core/unittest core)
#ml_add_test(lib/maths/common/unittest maths_common)
#ml_add_test(lib/maths/time_series/unittest maths_time_series)
#ml_add_test(lib/maths/analytics/unittest maths_analytics)
#ml_add_test(lib/model/unittest model)
#ml_add_test(lib/api/unittest api)
ml_add_test(lib/ver/unittest ver)
ml_add_test(lib/seccomp/unittest seccomp)
#ml_add_test(bin/controller/unittest controller)
#ml_add_test(bin/pytorch_inference/unittest pytorch_inference)

add_custom_target(build_tests
    DEPENDS ${ML_BUILD_TEST_DEPENDS}
)
add_custom_target(clean_test_results
    COMMAND ${CMAKE_COMMAND} -DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/test-clean-results.cmake
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_custom_target(run_tests
    DEPENDS clean_test_results ${ML_TEST_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_custom_target(test
    DEPENDS run_tests
    COMMAND ${CMAKE_COMMAND} -DTEST_DIR=${CMAKE_BINARY_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/test-check-success.cmake
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
