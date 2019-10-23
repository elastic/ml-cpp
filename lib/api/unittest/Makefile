#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=ml_test$(EXE_EXT)

USE_BOOST=1
USE_BOOST_FILESYSTEM_LIBS=1
USE_NET=1
USE_RAPIDJSON=1
USE_EIGEN=1

LIBS:=$(LIB_ML_MATHS) $(LIB_ML_MODEL) $(LIB_ML_API) $(LIB_ML_TEST)

all: build

SRCS=\
	Main.cc \
	CAnomalyJobLimitTest.cc \
	CAnomalyJobTest.cc \
	CBaseTokenListDataTyperTest.cc \
	CBoostedTreeInferenceModelBuilderTest.cc \
	CCategoryExamplesCollectorTest.cc \
	CConfigUpdaterTest.cc \
	CCsvInputParserTest.cc \
	CCsvOutputWriterTest.cc \
	CDataFrameAnalysisRunnerTest.cc \
	CDataFrameAnalysisSpecificationTest.cc \
	CDataFrameAnalyzerOutlierTest.cc \
	CDataFrameAnalyzerTrainingTest.cc \
	CDataFrameMockAnalysisRunner.cc \
	CDetectionRulesJsonParserTest.cc \
	CFieldConfigTest.cc \
	CFieldDataTyperTest.cc \
	CForecastRunnerTest.cc \
	CIoManagerTest.cc \
	CJsonOutputWriterTest.cc \
	CLengthEncodedInputParserTest.cc \
	CMemoryUsageEstimationResultJsonWriterTest.cc \
	CMockDataAdder.cc \
	CMockDataProcessor.cc \
	CMockSearcher.cc \
	CModelPlotDataJsonWriterTest.cc \
	CModelSnapshotJsonWriterTest.cc \
	CMultiFileDataAdderTest.cc \
	CNdJsonInputParserTest.cc \
	CNdJsonOutputWriterTest.cc \
	COutputChainerTest.cc \
	CPersistenceManagerTest.cc \
	CRestorePreviousStateTest.cc \
	CResultNormalizerTest.cc \
	CSingleStreamDataAdderTest.cc \
	CStateRestoreStreamFilterTest.cc \
	CStringStoreTest.cc \
	CTokenListDataTyperTest.cc \
	CTokenListReverseSearchCreatorTest.cc \


include $(CPP_SRC_HOME)/mk/stdcppunit.mk

