cmake_minimum_required(VERSION 3.7)

project("ML Model")

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
CAnnotatedProbability.cc
CAnnotatedProbabilityBuilder.cc
CAnomalyDetector.cc
CAnomalyDetectorModel.cc
CAnomalyDetectorModelConfig.cc
CAnomalyScore.cc
CBucketGatherer.cc
CCategoryExamplesCollector.cc
CCountingModel.cc
CCountingModelFactory.cc
CDataCategorizer.cc
CDataClassifier.cc
CDataGatherer.cc
CDetectionRule.cc
CDetectorEqualizer.cc
CDynamicStringIdRegistry.cc
CEventData.cc
CEventRateBucketGatherer.cc
CEventRateModel.cc
CEventRateModelFactory.cc
CEventRatePopulationModel.cc
CEventRatePopulationModelFactory.cc
CFeatureData.cc
CForecastDataSink.cc
CForecastModelPersist.cc
CGathererTools.cc
CHierarchicalResults.cc
CHierarchicalResultsAggregator.cc
CHierarchicalResultsNormalizer.cc
CHierarchicalResultsPopulator.cc
CHierarchicalResultsProbabilityFinalizer.cc
CIndividualModel.cc
CInterimBucketCorrector.cc
CLimits.cc
CLocalCategoryId.cc
CMemoryUsageEstimator.cc
CMetricBucketGatherer.cc
CMetricModel.cc
CMetricModelFactory.cc
CMetricPopulationModel.cc
CMetricPopulationModelFactory.cc
CAnnotation.cc
CModelDetailsView.cc
CModelFactory.cc
CModelPlotData.cc
CModelTools.cc
CMonitoredResource.cc
CPartitioningFields.cc
CPopulationModel.cc
CProbabilityAndInfluenceCalculator.cc
CResourceMonitor.cc
CRuleCondition.cc
CRuleScope.cc
CSample.cc
CSampleCounts.cc
CSearchKey.cc
CSimpleCountDetector.cc
CStringStore.cc
CTokenListCategory.cc
CTokenListDataCategorizerBase.cc
CTokenListReverseSearchCreator.cc
FrequencyPredicates.cc
FunctionTypes.cc
ModelTypes.cc
SModelParams.cc
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_library(MlModel SHARED ${SRCS} ${PLATFORM_SRCS})
target_link_libraries(MlModel ${Boost_LIBRARIES})
target_link_libraries(MlModel ${LOG4CXX_LIBRARIES})
target_link_libraries(MlModel MlCore)
target_link_libraries(MlModel MlMathsCommon)
target_link_libraries(MlModel MlMathsTimeSeries)


install(TARGETS MlModel DESTINATION lib)
