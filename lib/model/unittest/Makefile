#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=ml_test$(EXE_EXT)

USE_BOOST=1
USE_BOOST_FILESYSTEM_LIBS=1
USE_BOOST_TEST_LIBS=1
USE_RAPIDJSON=1
USE_EIGEN=1

LIBS:=$(LIB_ML_MATHS) $(LIB_ML_MODEL)

all: build

SRCS=\
	Main.cc \
	CAnnotatedProbabilityBuilderTest.cc \
	CAnomalyDetectorModelConfigTest.cc \
	CAnomalyScoreTest.cc \
	CBaseTokenListDataCategorizerTest.cc \
	CBucketQueueTest.cc \
	CCategoryExamplesCollectorTest.cc \
	CCountingModelTest.cc \
	CDetectionRuleTest.cc \
	CDetectorEqualizerTest.cc \
	CDynamicStringIdRegistryTest.cc \
	CEventRateAnomalyDetectorTest.cc \
	CEventRateDataGathererTest.cc \
	CEventRateModelTest.cc \
	CEventRatePopulationDataGathererTest.cc \
	CEventRatePopulationModelTest.cc \
	CForecastModelPersistTest.cc \
	CFunctionTypesTest.cc \
	CGathererToolsTest.cc \
	CHierarchicalResultsTest.cc \
	CHierarchicalResultsLevelSetTest.cc \
	CInterimBucketCorrectorTest.cc \
	CLimitsTest.cc \
	CMemoryUsageEstimatorTest.cc \
	CMetricAnomalyDetectorTest.cc \
	CMetricDataGathererTest.cc \
	CMetricModelTest.cc \
	CMetricPopulationDataGathererTest.cc \
	CMetricPopulationModelTest.cc \
	CModelDetailsViewTest.cc \
	CModelMemoryTest.cc \
	CModelToolsTest.cc \
	CModelTypesTest.cc \
	CProbabilityAndInfluenceCalculatorTest.cc \
	CResourceLimitTest.cc \
	CResourceMonitorTest.cc \
	CRuleConditionTest.cc \
	CSampleQueueTest.cc \
	CSearchKeyTest.cc \
	CStringStoreTest.cc \
	CTokenListDataCategorizerTest.cc \
	CTokenListReverseSearchCreatorTest.cc \
	CToolsTest.cc \
	Mocks.cc

include $(CPP_SRC_HOME)/mk/stdboosttest.mk

