project("ML Maths Time Series lib")

include_directories(../../../include)
include_directories(SYSTEM ../../../3rd_party/include)
include_directories(SYSTEM ../../../3rd_party/eigen)
include_directories(SYSTEM ../../../3rd_party/rapidjson/include)

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

add_library(MlMathsTimeSeries SHARED ${SRCS})
target_link_libraries(MlMathsTimeSeries PUBLIC ${Boost_LIBRARIES})
target_link_libraries(MlMathsTimeSeries PUBLIC MlMathsCommon)
target_link_libraries(MlMathsTimeSeries PUBLIC MlCore)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(MlMathsTimeSeries PRIVATE "-current_version ${ML_VERSION_NUM}" "-compatibility_version ${ML_VERSION_NUM}" "${COVERAGE}")
endif()

install(TARGETS MlMathsTimeSeries  DESTINATION lib)
