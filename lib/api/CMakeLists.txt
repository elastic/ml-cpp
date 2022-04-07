cmake_minimum_required(VERSION 3.7)

project("ML Api")

set (CMAKE_CXX_STANDARD 17)

add_compile_definitions(RAPIDJSON_HAS_STDSTRING DRAPIDJSON_NEON)

find_package(Boost 1.77.0 REQUIRED COMPONENTS iostreams filesystem program_options regex date_time log_setup thread unit_test_framework)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
add_compile_definitions(MacOSX)
set(Boost_COMPILER -clang-darwin13)
set(CMAKE_MACOSX_RPATH 1)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
set(Boost_COMPILER gcc10)
endif()

list( APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib )


include_directories(../../include)
include_directories(../../3rd_party/include)
include_directories(../../3rd_party/eigen)
include_directories(../../3rd_party/rapidjson/include)

set(SRCS
CAnnotationJsonWriter.cc
CAnomalyJob.cc
CAnomalyJobConfig.cc
CAnomalyJobConfigReader.cc
CBenchMarker.cc
CBoostedTreeInferenceModelBuilder.cc
CCategoryIdMapper.cc
CCmdSkeleton.cc
CConfigUpdater.cc
CCsvInputParser.cc
CCsvOutputWriter.cc
CDataFrameAnalysisConfigReader.cc
CDataFrameAnalysisInstrumentation.cc
CDataFrameAnalysisRunner.cc
CDataFrameAnalysisSpecification.cc
CDataFrameAnalysisSpecificationJsonWriter.cc
CDataFrameAnalyzer.cc
CDataFrameOutliersRunner.cc
CDataFrameTrainBoostedTreeClassifierRunner.cc
CDataFrameTrainBoostedTreeRegressionRunner.cc
CDataFrameTrainBoostedTreeRunner.cc
CDataProcessor.cc
CDetectionRulesJsonParser.cc
CFieldDataCategorizer.cc
CForecastRunner.cc
CGlobalCategoryId.cc
CHierarchicalResultsWriter.cc
CInferenceModelDefinition.cc
CInferenceModelMetadata.cc
CInputParser.cc
CIoManager.cc
CJsonOutputWriter.cc
CLengthEncodedInputParser.cc
CMemoryUsageEstimationResultJsonWriter.cc
CModelPlotDataJsonWriter.cc
CModelSizeStatsJsonWriter.cc
CModelSnapshotJsonWriter.cc
CNdInputParser.cc
CNdJsonInputParser.cc
CNdJsonOutputWriter.cc
CNoopCategoryIdMapper.cc
CPerPartitionCategoryIdMapper.cc
CPersistenceManager.cc
CResultNormalizer.cc
CSimpleOutputWriter.cc
CSingleFieldDataCategorizer.cc
CSingleStreamDataAdder.cc
CSingleStreamSearcher.cc
CStateRestoreStreamFilter.cc
ElasticsearchStateIndex.cc
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_library(MlApi SHARED ${SRCS} ${PLATFORM_SRCS})
target_link_libraries(MlApi ${Boost_LIBRARIES})
target_link_libraries(MlApi ${LOG4CXX_LIBRARIES})
target_link_libraries(MlApi MlCore)
target_link_libraries(MlApi MlMathsCommon)
target_link_libraries(MlApi MlMathsTimeSeries)
target_link_libraries(MlApi MlMathsAnalytics)
target_link_libraries(MlApi MlModel)

install(TARGETS MlApi DESTINATION lib)
