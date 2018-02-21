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

include $(CPP_SRC_HOME)/mk/stdtest.mk

# all the test cases this is used for use the Core library
# We keep the dependancies in here to a minimum so we can test
# libraries in isolation. Otherwise All libraries would be needed before
# we could test any.
REQUIRED_LIBS=$(LIB_ML_TEST) $(LIB_ML_CORE) $(LIB_ML_VER) $(CPPUNITLIBS)

LIBS:=$(LOCALLIBS) $(filter-out $(REQUIRED_LIBS), $(LIBS)) $(REQUIRED_LIBS)

CPPFLAGS+=$(INCLUDE_PATH)
LDFLAGS:=$(UTLDFLAGS) $(LDFLAGS) $(LIB_PATH) $(ML_VER_LDFLAGS)
PICFLAGS=$(PLATPIEFLAGS)

$(TARGET): $(OBJS) $(RESOURCE_FILE)
	$(CXX) $(LINK_OUT_FLAG)$@ $(PDB_FLAGS) $(OBJS) $(RESOURCE_FILE) $(LDFLAGS) $(LIBS)

build: $(TARGET)

clean:
	$(RM) $(OBJS_DIR)/*$(OBJECT_FILE_EXT) $(OBJS_DIR)/*.d* $(OBJS_DIR)/*.plist $(OBJS_DIR)/*.res $(APP_CLEAN) core core.* $(TARGET) $(basename $(TARGET)).pdb $(basename $(TARGET)).map $(basename $(TARGET)).exp cppunit_results.xml
	$(RMDIR) results
	$(RMDIR) data

relink:
	$(RM) $(TARGET)
	$(MAKE) -f $(filter-out %.mk %.d,$(MAKEFILE_LIST)) build

objcompile: $(OBJS)

analyze: $(ANALYZEOBJS)

