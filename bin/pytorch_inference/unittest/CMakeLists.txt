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

project("ML pytorch_inference unit tests")

include_directories(../../../include)
include_directories(SYSTEM ../../../3rd_party/include)
include_directories(SYSTEM ../../../3rd_party/eigen)
include_directories(SYSTEM ../../../3rd_party/rapidjson/include)

set (SRCS
    Main.cc
    CCommandParserTest.cc
    CThreadSettingsTest.cc
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(ml_test_pytorch_inference ${SRCS} $<TARGET_OBJECTS:MlPytorchInference>)

target_link_libraries(ml_test_pytorch_inference ${Boost_LIBRARIES_WITH_UNIT_TEST})
target_link_libraries(ml_test_pytorch_inference ${LIBXML2_LIBRARIES})
target_link_libraries(ml_test_pytorch_inference MlCore)
target_link_libraries(ml_test_pytorch_inference MlMathsCommon)
target_link_libraries(ml_test_pytorch_inference MlMathsAnalytics)
target_link_libraries(ml_test_pytorch_inference MlModel)
target_link_libraries(ml_test_pytorch_inference MlApi)
target_link_libraries(ml_test_pytorch_inference MlTest)
target_link_libraries(ml_test_pytorch_inference MlVer)
target_link_libraries(ml_test_pytorch_inference "${TORCH_LIB}")
target_link_libraries(ml_test_pytorch_inference "${C10_LIB}")

# we expect to be built out of source, in a sub directory
# so need to change working directory up a level to find
# test files
add_test(ml_test_pytorch_inference ml_test_pytorch_inference
)

# make "make test" run the test program
add_custom_target(test_pytorch_inference
     DEPENDS ml_test_pytorch_inference
     COMMAND ml_test_pytorch_inference
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
