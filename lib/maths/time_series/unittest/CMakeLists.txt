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

project("ML Maths Time Series unit tests")

include_directories(../../../../include)
include_directories(SYSTEM ../../../../3rd_party/include)
include_directories(SYSTEM ../../../../3rd_party/eigen)
include_directories(SYSTEM ../../../../3rd_party/rapidjson/include)

# TODO make platform safe
set (PLATFORM_SRCS
)

set (SRCS
Main.cc 
CAdaptiveBucketingTest.cc 
CCalendarComponentAdaptiveBucketingTest.cc 
CCalendarComponentTest.cc 
CCalendarCyclicTestTest.cc 
CCalendarFeatureTest.cc 
CCountMinSketchTest.cc 
CDecayRateControllerTest.cc 
CExpandingWindowTest.cc 
CForecastTest.cc 
CMathsMemoryTest.cc 
CSeasonalComponentAdaptiveBucketingTest.cc 
CSeasonalComponentTest.cc 
CSeasonalTimeTest.cc 
CSignalTest.cc 
CTimeSeriesDecompositionTest.cc 
CTimeSeriesModelTest.cc 
CTimeSeriesMultibucketFeaturesTest.cc 
CTimeSeriesSegmentationTest.cc 
CTimeSeriesTestForChangeTest.cc 
CTimeSeriesTestForSeasonalityTest.cc 
CTrendComponentTest.cc 
TestUtils.cc
)

add_executable(ml_test_maths_time_series ${SRCS} ${PLATFORM_SRCS})

target_link_libraries(ml_test_maths_time_series ${Boost_LIBRARIES})
target_link_libraries(ml_test_maths_time_series MlCore)
target_link_libraries(ml_test_maths_time_series MlMathsCommon)
target_link_libraries(ml_test_maths_time_series MlMathsTimeSeries)
target_link_libraries(ml_test_maths_time_series MlTest)


add_test(ml_test_maths_time_series ml_test_maths_time_series
)

add_custom_target(test_math_time_series
     DEPENDS ml_test_maths_time_series
     COMMAND ml_test_maths_time_series
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
