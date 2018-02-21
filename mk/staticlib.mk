#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
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
