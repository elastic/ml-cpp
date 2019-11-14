#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

ifndef RULES_DEFINED
RULES_DEFINED=1

# stop default SCCS get behaviour
%: s.%
%: SCCS/s.%

# generate a list of platform sources in this directory
# e.g. CTimeUtils_Linux.cc
LOCAL_OS_SRCS=$(wildcard *_$(OS).cc )

# transform source list to os specific (if exists)
# e.g. CTimeUtils.cc CHwAddress.cc
# ->   CTimeUtils_Linux.cc CHwAddress.cc
ifneq ($(LOCAL_OS_SRCS),)
# This is pretty inefficient but as the number of items
# in the loop is so small this shouldn't be an issue
# for(src $PLATFORM_SRCS)
#   for(local $LOCAL_OS_SRCS)
#       if(src ~= local)
#           file=local
#           break
#   if(file)
#       OS_SRCS += file
#   else
#       OS_SRCS += src
OS_SRCS=$(foreach src, $(PLATFORM_SRCS), $(if $(strip $(foreach local,$(LOCAL_OS_SRCS),$(if $(findstring $(basename $(src))_,$(local)),$(local),))), $(foreach local,$(LOCAL_OS_SRCS),$(if $(findstring $(basename $(src))_,$(local)),$(local),)), $(src)))
else
OS_SRCS=$(PLATFORM_SRCS)
endif

OBJS:=$(OBJS) $(patsubst %.cc, $(OBJS_DIR)/%$(OBJECT_FILE_EXT), $(SRCS))
ANALYZEOBJS:=$(ANALYZEOBJS) $(patsubst %.cc, $(OBJS_DIR)/%.xml, $(SRCS))

# define MAKE_PREFIX_SRC_PATH to prefix source file names with their path
# leave undefined for normal behaviour

$(OBJS_DIR)/%$(OBJECT_FILE_EXT): %.cc
	$(CXX) -c $(COMP_OUT_FLAG)$@ $(CXXFLAGS) $(PICFLAGS) $(PDB_FLAGS) $(CPPFLAGS) $(if $(MAKE_PREFIX_SRC_PATH), $(shell pwd)/)$<

$(OBJS_DIR)/%$(OBJECT_FILE_EXT): %.c
	$(CC) -c $(COMP_OUT_FLAG)$@ $(CFLAGS) $(PICFLAGS) $(PDB_FLAGS) $(CPPFLAGS) $(if $(MAKE_PREFIX_SRC_PATH), $(shell pwd)/)$<

$(OBJS_DIR)/%.plist: $(CPP_SRC_HOME)/gradle.properties $(CPP_SRC_HOME)/mk/make_info_plist.sh
	$(CPP_SRC_HOME)/mk/make_info_plist.sh "$(ML_APP_NAME)" false > $@

$(OBJS_DIR)/%.xml: %.cc
	$(CXX) $(ANALYZE_OUT_FLAG)$@ $(ANALYZEFLAGS) $(filter-out -DNDEBUG, $(CPPFLAGS)) $(if $(MAKE_PREFIX_SRC_PATH), $(shell pwd)/)$<

# JOB_NAME will be set for builds kicked off by Jenkins, where dependency
# checking is unnecessary as the Git clone is completely clean on starting the
# build
ifndef JOB_NAME

# This produces a set of dependencies for each source file.  These dependencies
# are wrapped in a GNU make wildcard variable so that if they cease to exist the
# wildcard will simply not find them.  (Before this was implemented, removal of
# a dependency would force a "make clean" to remove .d files referencing the
# removed dependencies.)  The assumption is that the last dependency will end
# in a letter whereas lines that shouldn't have a bracket appended will end in a
# backslash (line continuation).
$(OBJS_DIR)/%.d: %.cc
	@set -e; \
	rm -f $@; \
	echo "Finding dependencies of" $<; \
	$(CXX) $(CDEPFLAGS) $(INCLUDE_PATH) $(CPPFLAGS) $< $(DEP_FILTER) > $@.$$$$; \
	$(DEP_REFORMAT) < $@.$$$$ | sed 's/: /: $$(wildcard /' | sed 's~\([A-Za-z]\)$$~\1)~' > $@; \
	rm -f $@.$$$$

$(OBJS_DIR)/%.d: %.c
	@set -e; \
	rm -f $@; \
	echo "Finding dependencies of" $<; \
	$(CC) $(CDEPFLAGS) $(INCLUDE_PATH) $(CPPFLAGS) $< $(DEP_FILTER) > $@.$$$$; \
	$(DEP_REFORMAT) < $@.$$$$ | sed 's/: /: $$(wildcard /' | sed 's~\([A-Za-z]\)$$~\1)~' > $@; \
	rm -f $@.$$$$

DEPS=$(patsubst %.cc, $(OBJS_DIR)/%.d, $(SRCS))

ifneq ($(MAKECMDGOALS), clean)
ifdef DEPS
-include $(DEPS)
endif
endif

endif

$(OBJS_DIR)/%.res: $(CPP_SRC_HOME)/mk/%.rc $(CPP_SRC_HOME)/gradle.properties $(CPP_SRC_HOME)/mk/ml.ico $(CPP_SRC_HOME)/mk/make_rc_defines.sh
	$(RC) $(CPPFLAGS) $(shell $(CPP_SRC_HOME)/mk/make_rc_defines.sh $(notdir $(TARGET))) -Fo$@ $<

# Initialise includes and local libs
INCLUDE_PATH+=-I$(CPP_SRC_HOME)/include

# We link to the logging library by default, but very occasionally exclude it
ifndef NO_BOOST_LOG_LIBS
USE_BOOST=1
LDFLAGS+=$(BOOSTLOGLDFLAGS)
LOCALLIBS+=$(BOOSTLOGLIBS)
endif

ifdef USE_XML
INCLUDE_PATH+=$(XMLINCLUDES)
LDFLAGS+=$(XMLLIBLDFLAGS)
LOCALLIBS+=$(XMLLIBS)
endif

# if this uses BOOST add the paths
ifdef USE_BOOST
INCLUDE_PATH+=$(BOOSTINCLUDES)
CPPFLAGS+=$(BOOSTCPPFLAGS)
endif

# if this uses BOOST add the paths
ifdef USE_BOOST_LOGSETUP_LIBS
LDFLAGS+=$(BOOSTLOGSETUPLDFLAGS)
LOCALLIBS+=$(BOOSTLOGSETUPLIBS)
endif

# if this uses BOOST add the paths
ifdef USE_BOOST_REGEX_LIBS
LDFLAGS+=$(BOOSTREGEXLDFLAGS)
LOCALLIBS+=$(BOOSTREGEXLIBS)
endif

# if this uses BOOST add the paths
ifdef USE_BOOST_IOSTREAMS_LIBS
LDFLAGS+=$(BOOSTIOSTREAMSLDFLAGS)
LOCALLIBS+=$(BOOSTIOSTREAMSLIBS)
endif

# if this uses BOOST add the paths
ifdef USE_BOOST_PROGRAMOPTIONS_LIBS
LDFLAGS+=$(BOOSTPROGRAMOPTIONSLDFLAGS)
LOCALLIBS+=$(BOOSTPROGRAMOPTIONSLIBS)
endif

# if this uses BOOST add the paths
ifdef USE_BOOST_THREAD_LIBS
LDFLAGS+=$(BOOSTTHREADLDFLAGS)
LOCALLIBS+=$(BOOSTTHREADLIBS)
endif

# if this uses BOOST add the paths
ifdef USE_BOOST_FILESYSTEM_LIBS
LDFLAGS+=$(BOOSTFILESYSTEMLDFLAGS)
LOCALLIBS+=$(BOOSTFILESYSTEMLIBS)
endif

# if this uses BOOST add the paths
ifdef USE_BOOST_DATETIME_LIBS
LDFLAGS+=$(BOOSTDATETIMELDFLAGS)
LOCALLIBS+=$(BOOSTDATETIMELIBS)
endif

# if this uses BOOST add the paths
ifdef USE_BOOST_TEST_LIBS
LDFLAGS+=$(BOOSTTESTLDFLAGS)
LOCALLIBS+=$(BOOSTTESTLIBS)
endif

ifdef USE_RAPIDJSON
INCLUDE_PATH+=$(RAPIDJSONINCLUDES)
CPPFLAGS+=$(RAPIDJSONCPPFLAGS)
endif

ifdef USE_EIGEN
INCLUDE_PATH+=$(EIGENINCLUDES)
CPPFLAGS+=$(EIGENCPPFLAGS)
endif

# if this uses the Java Native Interface add paths
ifdef USE_JNI
INCLUDE_PATH+=$(JAVANATIVEINCLUDES)
LDFLAGS+=$(JAVANATIVELDFLAGS)
LOCALLIBS+=$(JAVANATIVELIBS)
endif

# if this uses ZLIB add the paths
ifdef USE_ZLIB
LDFLAGS+=$(ZLIBLDFLAGS)
LOCALLIBS+=$(ZLIBLIBS)
endif

# if this uses strptime() add the paths
ifdef USE_STRPTIME
LOCALLIBS+=$(STRPTIMELIBS)
endif

# if this uses network functionality add the paths
ifdef USE_NET
LOCALLIBS+=$(NETLIBS)
endif


endif

