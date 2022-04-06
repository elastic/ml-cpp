cmake_minimum_required(VERSION 3.7)

project("ML Api")

add_compile_definitions(RAPIDJSON_HAS_STDSTRING DRAPIDJSON_NEON MacOSX)

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

add_library(MlApi SHARED ${SRCS} ${PLATFORM_SRCS})
target_link_libraries(MlApi ${Boost_LIBRARIES})
target_link_libraries(MlApi ${LOG4CXX_LIBRARIES})
target_link_libraries(MlApi MlCore)
target_link_libraries(MlApi MlMathsCommon)
target_link_libraries(MlApi MlMathsTimeSeries)
target_link_libraries(MlApi MlMathsAnalytics)
target_link_libraries(MlApi MlModel)

install(TARGETS MlApi DESTINATION lib)
