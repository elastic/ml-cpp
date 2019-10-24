#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

include $(CPP_SRC_HOME)/mk/stdmodule.mk

LDFLAGS:=$(DYNAMICLIBLDFLAGS) $(LDFLAGS) $(LIB_PATH)
PICFLAGS=$(PLATPICFLAGS)

ifndef INSTALL_DIR
INSTALL_DIR=$(CPP_PLATFORM_HOME)/$(DYNAMIC_LIB_DIR)
endif

ifndef INSTALL_IMPORT_LIB_DIR
ifdef IMPORT_LIB_DIR
INSTALL_IMPORT_LIB_DIR=$(CPP_PLATFORM_HOME)/$(IMPORT_LIB_DIR)
IMPORT_LIB_NAME=$(basename $(TARGET))$(STATIC_LIB_EXT)
endif
endif

INSTALL_CMD=$(INSTALL) $(TARGET) $(INSTALL_DIR)
ifdef INSTALL_IMPORT_LIB_DIR
INSTALL_IMPORT_LIB_CMD=$(INSTALL) $(IMPORT_LIB_NAME) $(INSTALL_IMPORT_LIB_DIR)
endif
ifdef PDB_FLAGS
INSTALL_PDB_CMD=$(INSTALL) $(basename $(TARGET)).pdb $(INSTALL_DIR)
endif

$(TARGET): $(OBJS) $(RESOURCE_FILE)
	$(CXX) $(LINK_OUT_FLAG)$@ $(PDB_FLAGS) $(OBJS) $(RESOURCE_FILE) $(LDFLAGS) $(LOCALLIBS) $(LIBS)

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
	$(MKDIR) $(INSTALL_DIR)
	$(INSTALL_CMD)
ifdef INSTALL_PDB_CMD
	$(INSTALL_PDB_CMD)
endif
ifdef INSTALL_IMPORT_LIB_DIR
	$(MKDIR) $(INSTALL_IMPORT_LIB_DIR)
endif
ifdef INSTALL_IMPORT_LIB_CMD
	$(INSTALL_IMPORT_LIB_CMD)
endif

build: $(TARGET)
	$(MAKE) -f $(filter-out %.mk %.d,$(MAKEFILE_LIST)) install

clean:
	$(RM) $(OBJS_DIR)/*$(OBJECT_FILE_EXT) $(OBJS_DIR)/*.d* $(OBJS_DIR)/*.plist $(OBJS_DIR)/*.xml $(OBJS_DIR)/*.res $(APP_CLEAN) core core.* $(TARGET) $(IMPORT_LIB_NAME) $(basename $(TARGET)).pdb $(basename $(TARGET)).map $(basename $(TARGET)).exp
	+$(CLEAN_CMDS)
	$(RMDIR) results
