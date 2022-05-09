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

project("ML Seccomp unit tests")

include_directories(../../../include)

set (SRCS
Main.cc 
CSystemCallFilterTest.cc 
)

add_executable(ml_test_seccomp ${SRCS})

target_link_libraries(ml_test_seccomp ${Boost_LIBRARIES})
target_link_libraries(ml_test_seccomp ${LIBXML2_LIBRARIES})
target_link_libraries(ml_test_seccomp MlCore)
target_link_libraries(ml_test_seccomp MlSeccomp)
target_link_libraries(ml_test_seccomp MlTest)

add_test(ml_test_seccomp ml_test_seccomp)
# make "make test" run the test program
add_custom_target(test_seccomp
     DEPENDS ml_test_seccomp
     COMMAND ml_test_seccomp
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
