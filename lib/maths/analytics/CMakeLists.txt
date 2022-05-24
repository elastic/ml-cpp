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

project("ML Maths Analytics")

include_directories(../../../include)
include_directories(SYSTEM ../../../3rd_party/include)
include_directories(SYSTEM ../../../3rd_party/eigen)
include_directories(SYSTEM ../../../3rd_party/rapidjson/include)

add_compile_definitions(BUILDING_libMlMathsAnalytics)

set(SRCS
CBoostedTree.cc
CBoostedTreeFactory.cc
CBoostedTreeHyperparameters.cc
CBoostedTreeImpl.cc
CBoostedTreeLeafNodeStatistics.cc
CBoostedTreeLeafNodeStatisticsThreading.cc
CBoostedTreeLoss.cc
CBoostedTreeUtils.cc
CDataFrameCategoryEncoder.cc
CDataFramePredictiveModel.cc
CDataFrameUtils.cc
CMic.cc
COutliers.cc
CTreeShapFeatureImportance.cc
)

add_library(MlMathsAnalytics SHARED ${SRCS})
target_link_libraries(MlMathsAnalytics PUBLIC ${Boost_LIBRARIES})
target_link_libraries(MlMathsAnalytics PUBLIC MlMathsCommon)
target_link_libraries(MlMathsAnalytics PUBLIC MlCore)

add_dependencies(MlMathsAnalytics eigen)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(MlMathsAnalytics PRIVATE "-current_version ${ML_VERSION_NUM}" "-compatibility_version ${ML_VERSION_NUM}" "${COVERAGE}")
endif()

install(TARGETS MlMathsAnalytics  RUNTIME DESTINATION ${DYNAMIC_LIB_DIR} ARCHIVE DESTINATION ${IMPORT_LIB_DIR})
if(MSVC)
  set_property(TARGET MlMathsAnalytics PROPERTY COMPILE_PDB_NAME MlMathsAnalytics)
  install(FILES $<TARGET_PDB_FILE:MlMathsAnalytics> DESTINATION ${DYNAMIC_LIB_DIR} OPTIONAL)
endif()
