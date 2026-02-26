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
ml_add_test(lib/maths/common/unittest maths_common)
ml_add_test(lib/maths/time_series/unittest maths_time_series)
ml_add_test(lib/maths/analytics/unittest maths_analytics)
ml_add_test(lib/model/unittest model)
ml_add_test(lib/api/unittest api)
ml_add_test(lib/ver/unittest ver)
ml_add_test(lib/seccomp/unittest seccomp)
ml_add_test(bin/controller/unittest controller)
ml_add_test(bin/pytorch_inference/unittest pytorch_inference)

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
add_custom_target(run_tests_individually
        DEPENDS clean_test_results ${ML_TEST_INDIVIDUALLY_DEPENDS}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_custom_target(test
    DEPENDS run_tests
    COMMAND ${CMAKE_COMMAND} -DTEST_DIR=${CMAKE_BINARY_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/test-check-success.cmake
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_custom_target(test_individually
        DEPENDS run_tests_individually
        COMMAND ${CMAKE_COMMAND} -DTEST_DIR=${CMAKE_BINARY_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/test-check-success.cmake
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Unified parallel test target: discovers ALL test cases from ALL suites and
# runs them in a single CTest invocation, allowing CTest to optimally schedule
# across all available CPU cores. This replaces the sequential per-suite
# approach used by test_individually.
get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_isMultiConfig)
  set(_build_type_arg "-DBUILD_TYPE=$<CONFIG>")
else()
  set(_build_type_arg "")
endif()

add_custom_target(test_all_parallel
        DEPENDS build_tests
        COMMAND ${CMAKE_COMMAND}
            -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
            -DBUILD_DIR=${CMAKE_BINARY_DIR}
            ${_build_type_arg}
            -P ${CMAKE_SOURCE_DIR}/cmake/run-all-tests-parallel.cmake
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)