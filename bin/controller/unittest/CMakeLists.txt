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

project("ML controller unit tests")

include_directories(../../../include)
include_directories(SYSTEM ../../../3rd_party/include)
include_directories(SYSTEM ../../../3rd_party/eigen)
include_directories(SYSTEM ../../../3rd_party/rapidjson/include)


set (SRCS
    Main.cc
    CBlockingCallCancellingStreamMonitorTest.cc
    CCommandProcessorTest.cc
    CResponseJsonWriterTest.cc
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(ml_test_controller ${SRCS} ${PLATFORM_SRCS})

target_link_libraries(ml_test_controller ${CONTROLLER_OBJS})

target_link_libraries(ml_test_controller MlController)
target_link_libraries(ml_test_controller ${Boost_LIBRARIES})
target_link_libraries(ml_test_controller ${LIBXML2_LIBRARIES})
target_link_libraries(ml_test_controller MlCore)
target_link_libraries(ml_test_controller MlTest)
target_link_libraries(ml_test_controller MlVer)

# we expect to be built out of source, in a sub directory
# so need to change working directory up a level to find
# test files
add_test(ml_test_controller ml_test_controller
)

# make "make test" run the test program
add_custom_target(test_controller
     DEPENDS ml_test_controller
     COMMAND ml_test_controller
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
