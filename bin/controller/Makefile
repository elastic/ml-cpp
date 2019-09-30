#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=$(OBJS_DIR)/controller$(EXE_EXT)

INSTALL_DIR=$(CPP_PLATFORM_HOME)/$(EXE_DIR)

ML_LIBS=$(LIB_ML_CORE)

USE_BOOST=1
USE_BOOST_PROGRAMOPTIONS_LIBS=1

LIBS=$(ML_LIBS)

all:: build

SRCS= \
    Main.cc \
    CBlockingCallCancellerThread.cc \
    CCmdLineParser.cc \
    CCommandProcessor.cc \

ifneq ($(PLIST_FILE),)
all:: $(CPP_SRC_HOME)/gradle.properties $(CPP_SRC_HOME)/mk/make_info_plist.sh
	$(MKDIR) $(INSTALL_DIR)
	$(CPP_SRC_HOME)/mk/make_info_plist.sh "$(ML_APP_NAME)" true > $(INSTALL_DIR)/../$(notdir $(PLIST_FILE))
endif

include $(CPP_SRC_HOME)/mk/stdapp.mk

