#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=ml_test$(EXE_EXT)

USE_BOOST=1
USE_BOOST_PROGRAMOPTIONS_LIBS=1
USE_BOOST_TEST_LIBS=1
USE_RAPIDJSON=1

all: build

TESTED_OBJS=\
    ../.objs/C*$(OBJECT_FILE_EXT) \

SRCS=\
    Main.cc \
    CBlockingCallCancellingStreamMonitorTest.cc \
    CCommandProcessorTest.cc \
    CResponseJsonWriterTest.cc \

include $(CPP_SRC_HOME)/mk/stdboosttest.mk

