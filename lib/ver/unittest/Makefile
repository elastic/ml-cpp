#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=ml_test$(EXE_EXT)

USE_BOOST=1

all: build

SRCS=\
	Main.cc \
	CBuildInfoTest.cc \

include $(CPP_SRC_HOME)/mk/stdcppunit.mk

