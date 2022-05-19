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

add_subdirectory(../lib/core/unittest lib/core/unittest)
add_subdirectory(../lib/maths/common/unittest lib/maths/common/unittest)
add_subdirectory(../lib/maths/time_series/unittest lib/maths/time_series/unittest)
add_subdirectory(../lib/maths/analytics/unittest lib/maths/analytics/unittest)
add_subdirectory(../lib/model/unittest lib/model/unittest)
add_subdirectory(../lib/api/unittest lib/api/unittest)
add_subdirectory(../lib/ver/unittest lib/ver/unittest)
add_subdirectory(../lib/seccomp/unittest lib/seccomp/unittest)
add_subdirectory(../bin/controller/unittest bin/controller/unittest)
add_subdirectory(../bin/pytorch_inference/unittest bin/pytorch_inference/unittest)

add_custom_target(run_test_controller
    DEPENDS ml_test_controller
    COMMAND ml_test_controller
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/controller/unittest
)
add_custom_target(run_test_pytorch_inference
    DEPENDS ml_test_pytorch_inference
    COMMAND ml_test_pytorch_inference
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/pytorch_inference/unittest
)
add_custom_target(run_test_model
    DEPENDS ml_test_model
    COMMAND ml_test_model
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/model/unittest
)
add_custom_target(run_test_core
    DEPENDS ml_test_core
    COMMAND ml_test_core
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/core/unittest
)
add_custom_target(run_test_api
    DEPENDS ml_test_api
    COMMAND ml_test_api
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/api/unittest
)
add_custom_target(run_test_seccomp
    DEPENDS ml_test_seccomp
    COMMAND ml_test_seccomp
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/seccomp/unittest
)
add_custom_target(run_test_maths_common
    DEPENDS ml_test_maths_common
    COMMAND ml_test_maths_common
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/maths/common/unittest
)
add_custom_target(run_test_maths_time_series
    DEPENDS ml_test_maths_time_series
    COMMAND ml_test_maths_time_series
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/maths/time_series/unittest
)
add_custom_target(run_test_maths_analytics
    DEPENDS ml_test_maths_analytics
    COMMAND ml_test_maths_analytics
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/maths/analytics/unittest
)
add_custom_target(run_test_ver
    DEPENDS ml_test_ver
    COMMAND ml_test_ver
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/ver/unittest
)
add_custom_target(build_tests
    DEPENDS ml_test_model ml_test_core ml_test_api ml_test_seccomp ml_test_ver ml_test_maths_common ml_test_maths_time_series ml_test_maths_analytics ml_test_controller ml_test_pytorch_inference
)
add_custom_target(test
    DEPENDS run_test_model run_test_core run_test_api run_test_seccomp run_test_ver run_test_maths_common
    run_test_maths_time_series run_test_maths_analytics run_test_controller run_test_pytorch_inference
)
