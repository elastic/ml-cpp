cmake_minimum_required(VERSION 3.7)

project("ML Maths Analytics lib")

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

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_library(MlMathsAnalytics SHARED ${SRCS})
target_link_libraries(MlMathsAnalytics ${Boost_LIBRARIES})
target_link_libraries(MlMathsAnalytics MlMathsCommon)
target_link_libraries(MlMathsAnalytics MlCore)

install(TARGETS MlMathsAnalytics  DESTINATION lib)
