#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
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
