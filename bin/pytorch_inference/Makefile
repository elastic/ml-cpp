#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=$(OBJS_DIR)/pytorch_inference$(EXE_EXT)

INSTALL_DIR=$(CPP_PLATFORM_HOME)/$(EXE_DIR)

ML_LIBS=$(LIB_ML_CORE) $(LIB_ML_API)

USE_BOOST=1
USE_BOOST_PROGRAMOPTIONS_LIBS=1
USE_NET=1
USE_RAPIDJSON=1
USE_TORCH=1

# These lines can be removed once PyTorch code is updated to avoid using features
# that are deprecated in C++17 (and will be removed in C++20).
# See https://github.com/pytorch/pytorch/issues/69290
ifeq ($(OS),Windows)
    CPPFLAGS+=-D_HAS_FEATURES_REMOVED_IN_CXX20 -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
endif

LIBS=$(ML_LIBS)

all: build

SRCS= \
    Main.cc \
    CBufferedIStreamAdapter.cc \
    CCmdLineParser.cc \
    CCommandParser.cc \
    SettingsValidator.cc \

include $(CPP_SRC_HOME)/mk/stdapp.mk

