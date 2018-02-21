#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=$(OBJS_DIR)/categorize$(EXE_EXT)

INSTALL_DIR=$(CPP_PLATFORM_HOME)/bin

ML_LIBS=$(LIB_ML_CORE) $(LIB_ML_MATHS) $(LIB_ML_MODEL) $(LIB_ML_API)

USE_BOOST=1
USE_BOOST_PROGRAMOPTIONS_LIBS=1
USE_RAPIDJSON=1

LIBS=$(ML_LIBS)

all: build

SRCS= \
    Main.cc \
    CCmdLineParser.cc \

NO_TEST_CASES=1

include $(CPP_SRC_HOME)/mk/stdapp.mk

