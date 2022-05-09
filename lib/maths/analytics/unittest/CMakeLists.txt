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

project("ML Maths Analytics unit tests")

include_directories(../../../../include)
include_directories(SYSTEM ../../../../3rd_party/include)
include_directories(SYSTEM ../../../../3rd_party/eigen)
include_directories(SYSTEM ../../../../3rd_party/rapidjson/include)


set (SRCS
Main.cc 
CBoostedTreeLeafNodeStatisticsTest.cc 
CBoostedTreeLossTest.cc 
CBoostedTreeTest.cc 
CDataFrameCategoryEncoderTest.cc 
CDataFrameUtilsTest.cc 
CMicTest.cc 
COutliersTest.cc 
CTreeShapFeatureImportanceTest.cc 
TestUtils.cc 
)

add_executable(ml_test_maths_analytics ${SRCS})

target_link_libraries(ml_test_maths_analytics ${Boost_LIBRARIES})
target_link_libraries(ml_test_maths_analytics MlCore)
target_link_libraries(ml_test_maths_analytics MlMathsCommon)
target_link_libraries(ml_test_maths_analytics MlMathsAnalytics)
target_link_libraries(ml_test_maths_analytics MlTest)


# we expect to be built out of source, in a sub directory
# so need to change working directory up a level to find
# test files
add_test(ml_test_maths_analytics ml_test_maths_analytics
)

# make "make test" run the test program
add_custom_target(test_maths_analytics
     DEPENDS ml_test_maths_analytics
     COMMAND ml_test_maths_analytics
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
