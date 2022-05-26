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

set(ML_LINK_LIBRARIES
    ${Boost_LIBRARIES_WITH_UNIT_TEST}
    ${LIBXML2_LIBRARIES}
    MlCore
    MlMathsCommon
    MlMathsTimeSeries
    MlModel
    MlTest
)

ml_add_test_executable(model ${SRCS})
