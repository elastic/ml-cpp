#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

include $(CPP_SRC_HOME)/mk/stdmodule.mk

# This program is designed to run from the directory it's built in like a unit
# test program, hence UTLDFLAGS rather than EXELDFLAGS
LDFLAGS:=$(UTLDFLAGS) $(LDFLAGS) $(LIB_PATH) $(ML_VER_LDFLAGS)
PICFLAGS=$(PLATPIEFLAGS)
LIBS:=$(LOCALLIBS) $(LIB_ML_VER) $(LIBS)

$(TARGET): $(OBJS) $(RESOURCE_FILE) $(PLIST_FILE)
	$(CXX) $(LINK_OUT_FLAG)$@ $(PDB_FLAGS) $(OBJS) $(RESOURCE_FILE) $(LDFLAGS) $(PLIST_FILE_LDFLAGS) $(LIBS)

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
	@ echo "$(TARGET) IS NOT INSTALLED"

build: $(TARGET)
	$(MAKE) -f $(filter-out %.mk %.d,$(MAKEFILE_LIST)) install

clean:
	$(RM) $(OBJS_DIR)/*$(OBJECT_FILE_EXT) $(OBJS_DIR)/*.d* $(OBJS_DIR)/*.plist $(OBJS_DIR)/*.res $(APP_CLEAN) core core.* $(TARGET) $(basename $(TARGET)).pdb $(basename $(TARGET)).map $(basename $(TARGET)).exp
	+$(CLEAN_CMDS)
	$(RMDIR) results

