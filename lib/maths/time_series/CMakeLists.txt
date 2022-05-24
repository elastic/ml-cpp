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

project("ML Maths Time Series")

include_directories(../../../include)
include_directories(SYSTEM ../../../3rd_party/include)
include_directories(SYSTEM ../../../3rd_party/eigen)
include_directories(SYSTEM ../../../3rd_party/rapidjson/include)

add_compile_definitions(BUILDING_libMlMathsTimeSeries)

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

add_dependencies(MlMathsTimeSeries eigen)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(MlMathsTimeSeries PRIVATE "-current_version ${ML_VERSION_NUM}" "-compatibility_version ${ML_VERSION_NUM}" "${COVERAGE}")
endif()

install(TARGETS MlMathsTimeSeries  RUNTIME DESTINATION ${DYNAMIC_LIB_DIR} ARCHIVE DESTINATION ${IMPORT_LIB_DIR})
if(MSVC)
  set_property(TARGET MlMathsTimeSeries PROPERTY COMPILE_PDB_NAME MlMathsTimeSeries)
  install(FILES $<TARGET_PDB_FILE:MlMathsTimeSeries> DESTINATION ${DYNAMIC_LIB_DIR} OPTIONAL)
endif()
