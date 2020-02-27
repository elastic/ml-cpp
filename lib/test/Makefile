#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
include $(CPP_SRC_HOME)/mk/defines.mk

USE_BOOST=1
USE_BOOST_FILESYSTEM_LIBS=1
USE_BOOST_TEST_LIBS=1
USE_EIGEN=1
USE_RAPIDJSON=1

LIBS:=$(LIB_ML_CORE) $(LIB_ML_MATHS) $(LIB_ML_API)

PLATFORM_SRCS= \
    CShellCmdEscape.cc \
    CTestTmpDir.cc \

SRCS= \
    $(OS_SRCS) \
    CBoostTestXmlOutput.cc \
    CDataFrameAnalysisSpecificationFactory.cc \
    CDataFrameAnalyzerTrainingFactory.cc \
    CMultiFileDataAdder.cc \
    CMultiFileSearcher.cc \
    CRandomNumbers.cc \
    CTestObserver.cc \
    CThreadDataReader.cc \
    CThreadDataWriter.cc \
    CTimeSeriesTestData.cc \

TARGET=$(OBJS_DIR)/libMlTest$(DYNAMIC_LIB_EXT)

all: build

NO_TEST_CASES=1

include $(CPP_SRC_HOME)/mk/dynamiclib.mk
