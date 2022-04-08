project("ML Maths Analytics lib")

include_directories(../../../include)
include_directories(SYSTEM ../../../3rd_party/include)
include_directories(SYSTEM ../../../3rd_party/eigen)
include_directories(SYSTEM ../../../3rd_party/rapidjson/include)

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

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(MlMathsAnalytics PRIVATE "-current_version ${ML_VERSION_NUM}" "-compatibility_version ${ML_VERSION_NUM}" "${COVERAGE}")
endif()

install(TARGETS MlMathsAnalytics  DESTINATION lib)
