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

TARGET=$(OBJS_DIR)/autoconfig$(EXE_EXT)

INSTALL_DIR=$(CPP_PLATFORM_HOME)/bin

ML_LIBS=$(LIB_ML_CORE) $(LIB_ML_MATHS) $(LIB_ML_MODEL) $(LIB_ML_API) $(LIB_ML_CONFIG)

USE_BOOST=1
USE_BOOST_PROGRAMOPTIONS_LIBS=1
USE_EIGEN=1

LIBS=$(ML_LIBS)

all: build

SRCS= \
    Main.cc \
    CCmdLineParser.cc \

NO_TEST_CASES=1

include $(CPP_SRC_HOME)/mk/stdapp.mk

