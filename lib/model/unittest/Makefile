#
# ELASTICSEARCH CONFIDENTIAL
#
# Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
#
# Notice: this software, and all information contained
# therein, is the exclusive property of Elasticsearch BV
# and its licensors, if any, and is protected under applicable
# domestic and foreign law, and international treaties.
#
# Reproduction, republication or distribution without the
# express written consent of Elasticsearch BV is
# strictly prohibited.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=ml_test$(EXE_EXT)

USE_BOOST=1
USE_BOOST_FILESYSTEM_LIBS=1
USE_RAPIDJSON=1
USE_EIGEN=1

LIBS:=$(LIB_ML_MATHS) $(LIB_ML_MODEL)

all: build

SRCS=\
	Main.cc \
	CAnnotatedProbabilityBuilderTest.cc \
	CAnomalyDetectorModelConfigTest.cc \
	CAnomalyScoreTest.cc \
	CBucketQueueTest.cc \
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
	CStringStoreTest.cc \
	CToolsTest.cc \
	Mocks.cc

include $(CPP_SRC_HOME)/mk/stdcppunit.mk

