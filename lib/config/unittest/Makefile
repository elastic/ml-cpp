#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=ml_test$(EXE_EXT)

USE_BOOST=1
USE_BOOST_TEST_LIBS=1
USE_EIGEN=1

LIBS:=$(LIB_ML_CORE) $(LIB_ML_MATHS) $(LIB_ML_API) $(LIB_ML_CONFIG)

all: build

SRCS=\
Main.cc \
CAutoconfigurerParamsTest.cc \
CDataSemanticsTest.cc \
CDataSummaryStatisticsTest.cc \
CDetectorEnumeratorTest.cc \
CReportWriterTest.cc \

include $(CPP_SRC_HOME)/mk/stdboosttest.mk
