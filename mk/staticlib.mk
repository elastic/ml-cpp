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

include $(CPP_SRC_HOME)/mk/stdmodule.mk

LDFLAGS:=$(LDFLAGS)
PICFLAGS=$(PLATPICFLAGS)

$(TARGET): $(OBJS)
	$(AR) $(AR_OUT_FLAG)$@ $(OBJS) $(LDFLAGS) $(LIBS)

test:
	+$(TEST_CMDS)

testobjcompile:
	+$(TEST_OBJ_COMPILE_CMDS)

objcompile: $(OBJS) testobjcompile

analyze: $(ANALYZEOBJS)

relink:
	$(RM) $(TARGET)
	$(RM) $(INSTALL_DIR)/$(notdir $(TARGET))
	$(MAKE) -f $(filter-out %.mk %.d,$(MAKEFILE_LIST)) build
	+$(RELINK_CMDS)

install:

build: $(TARGET)

clean:
	$(RM) $(OBJS_DIR)/*$(OBJECT_FILE_EXT) $(OBJS_DIR)/*.d* $(OBJS_DIR)/*.plist $(APP_CLEAN) core core.* $(TARGET) $(basename $(TARGET)).pdb
	+$(CLEAN_CMDS)
