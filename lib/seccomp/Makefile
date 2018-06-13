#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=$(OBJS_DIR)/libMlSeccomp$(STATIC_LIB_EXT)


all: build

PLATFORM_SRCS= \
CSystemCallFilter.cc \

SRCS= \
$(OS_SRCS)

include $(CPP_SRC_HOME)/mk/staticlib.mk
