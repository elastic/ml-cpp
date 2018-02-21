#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=vfprog$(EXE_EXT)

INCLUDE_PATH=-I$(CPP_SRC_HOME)/devinclude

ML_LIBS=$(LIB_ML_CORE)

ifeq ($(OS),Windows)
ML_LIBS+= libVf.lib
else
ML_LIBS+= -lVf
endif

LIBS=$(ML_LIBS)

all: build

SRCS= \
    Main.cc \
    CIncrementer.cc \
    CLooper.cc \

NO_TEST_CASES=1

include $(CPP_SRC_HOME)/mk/stddevapp.mk

