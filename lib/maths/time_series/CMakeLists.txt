cmake_minimum_required(VERSION 3.7)

project("ML Maths Time Series lib")

set (CMAKE_CXX_STANDARD 17)

find_package(Boost 1.77.0 REQUIRED COMPONENTS iostreams filesystem program_options regex date_time log_setup thread unit_test_framework)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

add_compile_definitions(RAPIDJSON_HAS_STDSTRING DRAPIDJSON_NEON)
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
add_compile_definitions(MacOSX)
set(Boost_COMPILER -clang-darwin13)
set(CMAKE_MACOSX_RPATH 1)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
set(Boost_COMPILER gcc10)
endif()

list( APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib )


include_directories(../../../include)
include_directories(../../../3rd_party/include)
include_directories(../../../3rd_party/eigen)
include_directories(../../../3rd_party/rapidjson/include)

set(SRCS
CAdaptiveBucketing.cc 
CCalendarComponent.cc 
CCalendarComponentAdaptiveBucketing.cc 
CCalendarCyclicTest.cc 
CCalendarFeature.cc 
CCountMinSketch.cc 
CDecayRateController.cc 
CDecompositionComponent.cc 
CExpandingWindow.cc 
CModelStateSerialiser.cc 
CSeasonalComponent.cc 
CSeasonalComponentAdaptiveBucketing.cc 
CSeasonalTime.cc 
CSignal.cc 
CTimeSeriesDecomposition.cc 
CTimeSeriesDecompositionDetail.cc 
CTimeSeriesDecompositionStateSerialiser.cc 
CTimeSeriesDecompositionStub.cc 
CTimeSeriesModel.cc 
CTimeSeriesMultibucketFeatureSerialiser.cc 
CTimeSeriesSegmentation.cc 
CTimeSeriesTestForChange.cc 
CTimeSeriesTestForSeasonality.cc 
CTrendComponent.cc 
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_library(MlMathsTimeSeries SHARED ${SRCS})
target_link_libraries(MlMathsTimeSeries ${Boost_LIBRARIES})
target_link_libraries(MlMathsTimeSeries MlMathsCommon)
target_link_libraries(MlMathsTimeSeries MlCore)

install(TARGETS MlMathsTimeSeries  DESTINATION lib)
