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

project("ML Model unit tests")

include_directories(../../../include)
include_directories(SYSTEM ../../../3rd_party/include)
include_directories(SYSTEM ../../../3rd_party/eigen)
include_directories(SYSTEM ../../../3rd_party/rapidjson/include)

# TODO make platform safe
set (PLATFORM_SRCS
)

set (SRCS
Main.cc 
CAnnotatedProbabilityBuilderTest.cc 
CAnomalyDetectorModelConfigTest.cc 
CAnomalyScoreTest.cc 
CBucketQueueTest.cc 
CCategoryExamplesCollectorTest.cc 
CCountingModelTest.cc 
CDetectionRuleTest.cc 
CDetectorEqualizerTest.cc 
CDynamicStringIdRegistryTest.cc 
CEventRateAnomalyDetectorTest.cc 
CEventRateDataGathererTest.cc 
CEventRateModelTest.cc 
CEventRatePopulationDataGathererTest.cc 
CEventRatePopulationModelTest.cc 
CForecastModelPersistTest.cc 
CFunctionTypesTest.cc 
CGathererToolsTest.cc 
CHierarchicalResultsTest.cc 
CHierarchicalResultsLevelSetTest.cc 
CInterimBucketCorrectorTest.cc 
CLimitsTest.cc 
CLocalCategoryIdTest.cc 
CMemoryUsageEstimatorTest.cc 
CMetricAnomalyDetectorTest.cc 
CMetricDataGathererTest.cc 
CMetricModelTest.cc 
CMetricPopulationDataGathererTest.cc 
CMetricPopulationModelTest.cc 
CModelDetailsViewTest.cc 
CModelMemoryTest.cc 
CModelTestFixtureBase.cc 
CModelToolsTest.cc 
CModelTypesTest.cc 
CProbabilityAndInfluenceCalculatorTest.cc 
CResourceLimitTest.cc 
CResourceMonitorTest.cc 
CRuleConditionTest.cc 
CSampleQueueTest.cc 
CSearchKeyTest.cc 
CStringStoreTest.cc 
CTokenListCategoryTest.cc 
CTokenListDataCategorizerBaseTest.cc 
CTokenListDataCategorizerTest.cc 
CTokenListReverseSearchCreatorTest.cc 
CToolsTest.cc 
Mocks.cc
)

add_executable(ml_test_model ${SRCS} ${PLATFORM_SRCS})

target_link_libraries(ml_test_model ${Boost_LIBRARIES})
target_link_libraries(ml_test_model libMlCore)
target_link_libraries(ml_test_model libMlMathsCommon)
target_link_libraries(ml_test_model libMlMathsTimeSeries)
target_link_libraries(ml_test_model libMlModel)
target_link_libraries(ml_test_model libMlTest)
target_link_libraries(ml_test_model ${NETLIBS})


add_test(ml_test_model ml_test_model
)

# make "make test" run the test program
# we expect to be built out of source, in a sub ditrectory
# so need to change working directory up a level to find
# test files
add_custom_target(test_model
     DEPENDS ml_test_model
     COMMAND ml_test_model
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
