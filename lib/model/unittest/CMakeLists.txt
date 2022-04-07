cmake_minimum_required(VERSION 3.7)

project("ML Model lib unit tests")

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

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(ml_test_model ${SRCS} ${PLATFORM_SRCS})

target_link_libraries(ml_test_model ${Boost_LIBRARIES})
target_link_libraries(ml_test_model MlCore)
target_link_libraries(ml_test_model MlMathsCommon)
target_link_libraries(ml_test_model MlMathsTimeSeries)
target_link_libraries(ml_test_model MlModel)
target_link_libraries(ml_test_model MlTest)
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
