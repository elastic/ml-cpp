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

TARGET=$(OBJS_DIR)/libVf$(DYNAMIC_LIB_EXT)

INCLUDE_PATH=-I$(CPP_SRC_HOME)/devinclude

all: build

SRCS= \
CIncrementer.cc \
CLooper.cc \

NO_TEST_CASES=1

include $(CPP_SRC_HOME)/mk/dynamiclib.mk
