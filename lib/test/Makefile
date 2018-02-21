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

USE_XML=1
USE_BOOST=1
USE_BOOST_FILESYSTEM_LIBS=1
USE_CPPUNIT=1
USE_EIGEN=1

LIBS:=$(LIB_ML_CORE) $(LIB_ML_MATHS)

PLATFORM_SRCS= \
    CShellCmdEscape.cc \
    CTestTmpDir.cc \

SRCS= \
    $(OS_SRCS) \
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
