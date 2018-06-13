#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

include $(CPP_SRC_HOME)/mk/stdmodule.mk

LDFLAGS:=$(EXELDFLAGS) $(LDFLAGS) $(LIB_PATH) $(ML_VER_LDFLAGS) $(ML_SECCOMP_LDFLAGS)
PICFLAGS=$(PLATPIEFLAGS)
LIBS:=$(LOCALLIBS) $(LIB_ML_VER) $(LIB_ML_SECCOMP) $(LIBS)

ifndef INSTALL_DIR
INSTALL_DIR=$(CPP_PLATFORM_HOME)/bin
endif

ifndef CONF_INSTALL_DIR
ifdef TARGET_CONF
CONF_INSTALL_DIR=$(CPP_DISTRIBUTION_HOME)/resources
endif
endif

INSTALL_CMD=$(MKDIR) $(INSTALL_DIR); $(INSTALL) $(TARGET) $(INSTALL_DIR)
ifdef PDB_FLAGS
INSTALL_PDB_CMD=$(INSTALL) $(basename $(TARGET)).pdb $(INSTALL_DIR)
endif
ifdef TARGET_CONF
#CONF_INSTALL_CMD=$(MKDIR) $(CONF_INSTALL_DIR); $(INSTALL) $(TARGET_CONF) $(CONF_INSTALL_DIR)
endif

$(TARGET): $(OBJS) $(RESOURCE_FILE)
	$(CXX) $(LINK_OUT_FLAG)$@ $(PDB_FLAGS) $(OBJS) $(RESOURCE_FILE) $(LDFLAGS) $(LIBS)

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
	$(INSTALL_CMD)
ifdef INSTALL_PDB_CMD
	$(INSTALL_PDB_CMD)
endif
	$(CONF_INSTALL_CMD)

build: $(TARGET)
	$(MAKE) -f $(filter-out %.mk %.d,$(MAKEFILE_LIST)) install

clean:
	$(RM) $(OBJS_DIR)/*$(OBJECT_FILE_EXT) $(OBJS_DIR)/*.d* $(OBJS_DIR)/*.plist $(OBJS_DIR)/*.res $(APP_CLEAN) core core.* $(TARGET) $(basename $(TARGET)).pdb $(basename $(TARGET)).map $(basename $(TARGET)).exp
	+$(CLEAN_CMDS)
	$(RMDIR) results

