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

project("ML Model")

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/eigen)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)

add_compile_definitions(BUILDING_libMlModel)

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
target_link_libraries(MlModel PUBLIC ${Boost_LIBRARIES})
target_link_libraries(MlModel PUBLIC MlCore)
target_link_libraries(MlModel PUBLIC MlMathsCommon)
target_link_libraries(MlModel PUBLIC MlMathsTimeSeries)

add_dependencies(MlModel eigen)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(MlModel PRIVATE "-current_version ${ML_VERSION_NUM}" "-compatibility_version ${ML_VERSION_NUM}" "${COVERAGE}")
endif()


install(TARGETS MlModel RUNTIME DESTINATION ${DYNAMIC_LIB_DIR} ARCHIVE DESTINATION ${IMPORT_LIB_DIR})
if(MSVC)
  install(FILES $<TARGET_PDB_FILE:MlModel> DESTINATION ${DYNAMIC_LIB_DIR} OPTIONAL)
endif()
