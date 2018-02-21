#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

include $(CPP_SRC_HOME)/mk/rules.mk

# We use TOP_DIR_MKF_FIRST and TOP_DIR_MKF_LAST to allow a directory to contain
# both sub-directories and source code:
# - The "normal" Makefile is used to recurse into the sub-directories
# - TOP_DIR_MKF_FIRST is used to perform actions at this level BEFORE
#   recursing into the sub-directories
# - TOP_DIR_MKF_LAST is used to perform actions at this level AFTER
#   recursing into the sub-directories
#
# If $ML_DEBUG is set then the recursion will stop at the first error;
# otherwise it will attempt to build every directory even after an earlier
# one fails.  This latter behaviour is useful during nightly builds as it
# means each nightly build has a chance to uncover more than one error.

all:
ifdef TOP_DIR_MKF_FIRST
	@ $(MAKE) -f $(TOP_DIR_MKF_FIRST)
endif
	@ FAILED=0; \
	 for i in $(COMPONENTS) ; \
	 do \
	 echo "$(notdir $(MAKE)): Component=$$i, Target=$@, Time=`date`"; \
	 (cd $$i && $(MAKE)); \
	 if [ $$? -ne 0 ] ; then \
	   FAILED=1; \
	   if [ -z "$(ML_KEEP_GOING)" ]; then exit 1; fi; \
	 fi; \
	 done; \
	 exit $$FAILED
ifdef TOP_DIR_MKF_LAST
	@ $(MAKE) -f $(TOP_DIR_MKF_LAST)
endif

# The purpose of the objcompile target is to build all C++ object files from all
# directories in parallel (without trying to link them, because linking requires
# dependent binaries be built in order).  This is achieved by parallel recursion
# into sub-directories.  However, unfortunately, some of our directory names
# clash with make target names - at present "test" and "install" fall into this
# category.  To avoid these clashes we suffix the raw directory names in
# $(COMPONENTS) with /targetdirectory (and then later strip this when we need
# the directory name).
SUFFIXED_COMPONENTS=$(addsuffix /targetdirectory, $(COMPONENTS))

.PHONY: $(SUFFIXED_COMPONENTS)

# The intention is that only the objcompile target should depend on this
$(SUFFIXED_COMPONENTS)::
ifdef TOP_DIR_MKF_FIRST
	-@ $(MAKE) objcompile -f $(TOP_DIR_MKF_FIRST)
endif
	+@if ls $(dir $@)/*.cc > /dev/null 2>&1 || grep toplevel.mk $(dir $@i)/Makefile > /dev/null 2>&1 ; then \
	 $(MAKE) -C $(dir $@) objcompile ; \
	fi
ifdef TOP_DIR_MKF_LAST
	-@ $(MAKE) objcompile -f $(TOP_DIR_MKF_LAST)
endif

objcompile: $(SUFFIXED_COMPONENTS)

analyze:
ifdef TOP_DIR_MKF_FIRST
	@ $(MAKE) analyze -f $(TOP_DIR_MKF_FIRST)
endif
	@ FAILED=0; \
	 for i in $(COMPONENTS) ; \
	 do \
	 echo "$(notdir $(MAKE)): Component=$$i, Target=$@, Time=`date`"; \
	 (cd $$i && $(MAKE) analyze ); \
	 if [ $$? -ne 0 ] ; then \
	   FAILED=1; \
	   if [ -z "$(ML_KEEP_GOING)" ]; then exit 1; fi; \
	 fi; \
	 done; \
	 exit $$FAILED
ifdef TOP_DIR_MKF_LAST
	@ $(MAKE) analyze -f $(TOP_DIR_MKF_LAST)
endif

clean:
ifdef TOP_DIR_MKF_FIRST
	@ $(MAKE) clean -f $(TOP_DIR_MKF_FIRST)
endif
	@ for i in $(COMPONENTS) ; \
	 do \
	 echo "$(notdir $(MAKE)): Component=$$i, Target=$@, Time=`date`"; \
	 (cd $$i && $(MAKE) clean ); \
	 done
ifdef TOP_DIR_MKF_LAST
	@ $(MAKE) clean -f $(TOP_DIR_MKF_LAST)
endif

ifdef TOP_DIR_MKF_FIRST
FIRST_TEST_CMD=$(MAKE) test -f $(TOP_DIR_MKF_FIRST)
else
FIRST_TEST_CMD=true
endif
test:
	@ FAILED=0; \
	 $(FIRST_TEST_CMD) ; \
	 if [ $$? -ne 0 ] ; then \
	   FAILED=1; \
	   if [ -z "$(ML_KEEP_GOING)" ]; then exit 1; fi; \
	 fi; \
	 for i in $(COMPONENTS) ; \
	 do \
	 echo "$(notdir $(MAKE)): Component=$$i, Target=$@, Time=`date`"; \
	 (cd $$i && $(MAKE) test ); \
	 if [ $$? -ne 0 ] ; then \
	   FAILED=1; \
	   if [ -z "$(ML_KEEP_GOING)" ]; then exit 1; fi; \
	 fi; \
	 done; \
	 exit $$FAILED
ifdef TOP_DIR_MKF_LAST
	@ $(MAKE) test -f $(TOP_DIR_MKF_LAST)
endif

relink:
ifdef TOP_DIR_MKF_FIRST
	@ $(MAKE) relink -f $(TOP_DIR_MKF_FIRST)
endif
	@ FAILED=0; \
	 for i in $(COMPONENTS) ; \
	 do \
	 echo "$(notdir $(MAKE)): Component=$$i, Target=$@, Time=`date`"; \
	 (cd $$i && $(MAKE) relink ); \
	 if [ $$? -ne 0 ] ; then \
	   FAILED=1; \
	   if [ -z "$(ML_KEEP_GOING)" ]; then exit 1; fi; \
	 fi; \
	 done; \
	 exit $$FAILED
ifdef TOP_DIR_MKF_LAST
	@ $(MAKE) relink -f $(TOP_DIR_MKF_LAST)
endif

install:
ifdef TOP_DIR_MKF_FIRST
	@ $(MAKE) install -f $(TOP_DIR_MKF_FIRST)
endif
	@ FAILED=0; \
	 for i in $(COMPONENTS) ; \
	 do \
	 echo "$(notdir $(MAKE)): Component=$$i, Target=$@, Time=`date`"; \
	 (cd $$i && $(MAKE) install ); \
	 if [ $$? -ne 0 ] ; then \
	   FAILED=1; \
	   if [ -z "$(ML_KEEP_GOING)" ]; then exit 1; fi; \
	 fi; \
	 done; \
	 exit $$FAILED
ifdef TOP_DIR_MKF_LAST
	@ $(MAKE) install -f $(TOP_DIR_MKF_LAST)
endif

build:
ifdef TOP_DIR_MKF_FIRST
	@ $(MAKE) build -f $(TOP_DIR_MKF_FIRST)
endif
	@ set -e || exit 1 ; \
	 for i in $(COMPONENTS) ; \
	 do \
	 echo "$(notdir $(MAKE)): Component=$$i, Target=$@, Time=`date`"; \
	 (cd $$i && $(MAKE) build || exit 1 ); \
	 done
ifdef TOP_DIR_MKF_LAST
	@ $(MAKE) build -f $(TOP_DIR_MKF_LAST)
endif

