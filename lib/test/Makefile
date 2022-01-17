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
include $(CPP_SRC_HOME)/mk/defines.mk

USE_BOOST=1
USE_BOOST_FILESYSTEM_LIBS=1
USE_BOOST_TEST_LIBS=1
USE_EIGEN=1
USE_RAPIDJSON=1

LIBS:=$(LIB_ML_CORE) $(LIB_ML_MATHS_COMMON) $(LIB_ML_MATHS_ANALYTICS) $(LIB_ML_API)

PLATFORM_SRCS= \
    CShellCmdEscape.cc \

SRCS= \
    $(OS_SRCS) \
    CBoostTestXmlOutput.cc \
    CDataFrameAnalysisSpecificationFactory.cc \
    CDataFrameAnalyzerTrainingFactory.cc \
    CMultiFileDataAdder.cc \
    CMultiFileSearcher.cc \
    CProgramCounterClearingFixture.cc \
    CRandomNumbers.cc \
    CTestObserver.cc \
    CTestTmpDir.cc \
    CThreadDataReader.cc \
    CThreadDataWriter.cc \
    CTimeSeriesTestData.cc \

TARGET=$(OBJS_DIR)/libMlTest$(DYNAMIC_LIB_EXT)

all: build

NO_TEST_CASES=1

include $(CPP_SRC_HOME)/mk/dynamiclib.mk
