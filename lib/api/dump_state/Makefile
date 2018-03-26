#
# ELASTICSEARCH CONFIDENTIAL
#
# Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

TARGET=dump_state$(EXE_EXT)

ML_LIBS=$(LIB_ML_CORE) $(LIB_ML_MATHS) $(LIB_ML_MODEL) $(LIB_ML_API)

USE_BOOST=1
USE_BOOST_FILESYSTEM_LIBS=1
USE_RAPIDJSON=1
USE_EIGEN=1

LIBS=$(ML_LIBS)

all: build

SRCS= \
    Main.cc

NO_TEST_CASES=1

include $(CPP_SRC_HOME)/mk/stddevapp.mk

