cmake_minimum_required(VERSION 3.7)

project("ML Maths Time Series lib unit tests")

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

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(ml_test_maths_time_series ${SRCS} ${PLATFORM_SRCS})

target_link_libraries(ml_test_maths_time_series ${Boost_LIBRARIES})
target_link_libraries(ml_test_maths_time_series MlCore)
target_link_libraries(ml_test_maths_time_series MlMathsCommon)
target_link_libraries(ml_test_maths_time_series MlMathsTimeSeries)
target_link_libraries(ml_test_maths_time_series MlTest)


# we expect to be built out of source, in a sub ditrectory
# so need to change working directory up a level to find
# test files
add_test(ml_test_maths_time_series ml_test_maths_time_series
)

# make "make test" run the test program
add_custom_target(test_math_time_series
     DEPENDS ml_test_maths_time_series
     COMMAND ml_test_maths_time_series
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
