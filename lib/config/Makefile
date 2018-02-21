#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=$(OBJS_DIR)/libMlConfig$(DYNAMIC_LIB_EXT)

USE_BOOST=1
USE_EIGEN=1

LIBS:=$(LIB_ML_CORE) $(LIB_ML_MATHS) $(LIB_ML_MODEL) $(LIB_ML_API)

all: build

SRCS= \
CAutoconfigurer.cc \
CAutoconfigurerDetectorPenalties.cc \
CAutoconfigurerFieldRolePenalties.cc \
CAutoconfigurerParams.cc \
CDataCountStatistics.cc \
CDataSemantics.cc \
CDataSummaryStatistics.cc \
CDetectorEnumerator.cc \
CDetectorFieldRolePenalty.cc \
CDetectorRecord.cc \
CDetectorSpecification.cc \
CFieldRolePenalty.cc \
CFieldStatistics.cc \
CLongTailPenalty.cc \
CLowInformationContentPenalty.cc \
CLowVariationPenalty.cc \
Constants.cc \
CNotEnoughDataPenalty.cc \
ConfigTypes.cc \
CPenalty.cc \
CPolledDataPenalty.cc \
CReportWriter.cc \
CSpanTooSmallForBucketLengthPenalty.cc \
CSparseCountPenalty.cc \
CTools.cc \
CTooMuchDataPenalty.cc \

include $(CPP_SRC_HOME)/mk/dynamiclib.mk
