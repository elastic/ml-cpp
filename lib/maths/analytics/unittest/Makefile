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

TARGET=ml_test$(EXE_EXT)

USE_BOOST=1
USE_BOOST_FILESYSTEM_LIBS=1
USE_BOOST_TEST_LIBS=1
USE_RAPIDJSON=1
USE_EIGEN=1

LIBS:=$(LIB_ML_MATHS_COMMON) $(LIB_ML_MATHS_ANALYTICS)

all: build

SRCS=\
	Main.cc \
	BoostedTreeTestData.cc \
	CBoostedTreeHyperparametersTest.cc \
	CBoostedTreeLeafNodeStatisticsTest.cc \
	CBoostedTreeLossTest.cc \
	CBoostedTreeTest.cc \
	CBoostedTreeUtilsTest.cc \
	CDataFrameCategoryEncoderTest.cc \
	CDataFrameUtilsTest.cc \
	CMicTest.cc \
	COutliersTest.cc \
	CTreeShapFeatureImportanceTest.cc \
	TestUtils.cc \

include $(CPP_SRC_HOME)/mk/stdboosttest.mk
