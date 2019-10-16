#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
include $(CPP_SRC_HOME)/mk/defines.mk

USE_XML=1
USE_BOOST=1
USE_BOOST_FILESYSTEM_LIBS=1
USE_CPPUNIT=1
USE_EIGEN=1
USE_RAPIDJSON=1

LIBS:=$(LIB_ML_CORE) $(LIB_ML_MATHS)

PLATFORM_SRCS= \
    CShellCmdEscape.cc \
    CTestTmpDir.cc \

SRCS= \
    $(OS_SRCS) \
    CDataFrameAnalysisSpecificationFactory.cc \
    CMultiFileDataAdder.cc \
    CMultiFileSearcher.cc \
    CRandomNumbers.cc \
    CTestRunner.cc \
    CTestTimer.cc \
    CTimeSeriesTestData.cc \
    CTimingXmlOutputterHook.cc \

TARGET=$(OBJS_DIR)/libMlTest$(DYNAMIC_LIB_EXT)

all: build

NO_TEST_CASES=1

include $(CPP_SRC_HOME)/mk/dynamiclib.mk
