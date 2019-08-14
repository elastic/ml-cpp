#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

OS=MacOSX

CPP_PLATFORM_HOME=$(CPP_DISTRIBUTION_HOME)/platform/darwin-x86_64

CROSS_TARGET_PLATFORM=x86_64-apple-macosx10.11
SYSROOT=/usr/local/sysroot-$(CROSS_TARGET_PLATFORM)
CLANGVER=3.9
CROSS_FLAGS=--sysroot=$(SYSROOT) -B /usr/local/bin -target $(CROSS_TARGET_PLATFORM)
CC=clang-$(CLANGVER) $(CROSS_FLAGS)
CXX=clang++-$(CLANGVER) $(CROSS_FLAGS) -std=c++14 -stdlib=libc++

ifndef ML_DEBUG
OPTCFLAGS=-O3
OPTCPPFLAGS=-DNDEBUG -DEXCLUDE_TRACE_LOGGING
endif

ifdef ML_DEBUG
ifdef ML_COVERAGE
COVERAGE=--coverage
endif
endif

# Start by enabling all warnings and then disable the really pointless/annoying ones
CFLAGS=-g $(OPTCFLAGS) -msse4.2 -fstack-protector -Weverything -Werror-switch -Wno-deprecated -Wno-disabled-macro-expansion -Wno-documentation-deprecated-sync -Wno-documentation-unknown-command -Wno-float-equal -Wno-gnu -Wno-missing-prototypes -Wno-padded -Wno-sign-conversion -Wno-unreachable-code -Wno-used-but-marked-unused $(COVERAGE)
CXXFLAGS=$(CFLAGS) -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-exit-time-destructors -Wno-global-constructors -Wno-undefined-reinterpret-cast -Wno-unused-member-function -Wno-weak-vtables
CPPFLAGS=-isystem $(SYSROOT)/usr/include/c++/v1 -isystem $(CPP_SRC_HOME)/3rd_party/include -isystem $(SYSROOT)/usr/local/include -D$(OS) $(OPTCPPFLAGS)
ANALYZEFLAGS=--analyze
CDEPFLAGS=-MM
COMP_OUT_FLAG=-o
ANALYZE_OUT_FLAG=-o
LINK_OUT_FLAG=-o
DEP_REFORMAT=sed 's,\($*\)\.o[ :]*,$(OBJS_DIR)\/\1.o $@ : ,g'
LOCALLIBS=
NETLIBS=
BOOSTVER=1_65_1
BOOSTCLANGVER:=$(shell $(CXX) -dumpversion | awk -F. '{ print $$1$$2; }')
# Use -isystem instead of -I for Boost headers to suppress warnings from Boost
BOOSTINCLUDES=-isystem $(SYSROOT)/usr/local/include/boost-$(BOOSTVER)
BOOSTCPPFLAGS=-DBOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
BOOSTREGEXLIBS=-lboost_regex-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTVER)
BOOSTIOSTREAMSLIBS=-lboost_iostreams-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTVER)
BOOSTPROGRAMOPTIONSLIBS=-lboost_program_options-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTVER)
BOOSTTHREADLIBS=-lboost_thread-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTVER) -lboost_system-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTVER)
BOOSTFILESYSTEMLIBS=-lboost_filesystem-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTVER) -lboost_system-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTVER)
BOOSTDATETIMELIBS=-lboost_date_time-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTVER)
RAPIDJSONINCLUDES=-isystem $(CPP_SRC_HOME)/3rd_party/rapidjson/include
RAPIDJSONCPPFLAGS=-DRAPIDJSON_HAS_STDSTRING -DRAPIDJSON_SSE42
EIGENCPPFLAGS=-DEIGEN_MPL2_ONLY
XMLINCLUDES=-isystem $(SYSROOT)/usr/include/libxml2
XMLLIBLDFLAGS=-L$(SYSROOT)/usr/lib
XMLLIBS=-lxml2
ML_VERSION_NUM=$(shell cat $(CPP_SRC_HOME)/gradle.properties | grep '^elasticsearchVersion' | awk -F= '{ print $$2 }' | xargs echo | sed 's/-.*//')
DYNAMICLIBLDFLAGS=-current_version $(ML_VERSION_NUM) -compatibility_version $(ML_VERSION_NUM) -dynamiclib -Wl,-dead_strip_dylibs $(COVERAGE) -Wl,-install_name,@rpath/$(notdir $(TARGET)) -L$(CPP_PLATFORM_HOME)/lib -Wl,-rpath,@loader_path/. -Wl,-headerpad_max_install_names
CPPUNITLIBS=-lcppunit
LOG4CXXLIBS=-llog4cxx
ZLIBLIBS=-lz
EXELDFLAGS=-bind_at_load -L$(CPP_PLATFORM_HOME)/lib $(COVERAGE) -Wl,-rpath,@loader_path/../lib -Wl,-headerpad_max_install_names
UTLDFLAGS=$(EXELDFLAGS) -Wl,-rpath,$(CPP_PLATFORM_HOME)/lib
PLIST_FILE=$(OBJS_DIR)/Info.plist
PLIST_FILE_LDFLAGS=-Wl,-sectcreate,__TEXT,__info_plist,$(PLIST_FILE)
OBJECT_FILE_EXT=.o
DYNAMIC_LIB_EXT=.dylib
DYNAMIC_LIB_DIR=lib
STATIC_LIB_EXT=.a
SHELL_SCRIPT_EXT=.sh
UT_TMP_DIR=/tmp/$(LOGNAME)
LIB_ML_API=-lMlApi
LIB_ML_CORE=-lMlCore
LIB_ML_VER=-lMlVer
ML_VER_LDFLAGS=-L$(CPP_SRC_HOME)/lib/ver/.objs
LIB_ML_MATHS=-lMlMaths
LIB_ML_CONFIG=-lMlConfig
LIB_ML_MODEL=-lMlModel
LIB_ML_SECCOMP=-lMlSeccomp
ML_SECCOMP_LDFLAGS=-L$(CPP_SRC_HOME)/lib/seccomp/.objs
LIB_ML_TEST=-lMlTest

LIB_PATH+=-L$(SYSROOT)/usr/local/lib

# Using cp instead of install here, to avoid every file being given execute
# permissions
INSTALL=cp
CP=cp
MKDIR=mkdir -p
RM=rm -f
RMDIR=rm -rf
MV=mv -f
SED=sed
ECHO=echo
CAT=cat
LN=ln
AR=/usr/local/bin/$(CROSS_TARGET_PLATFORM)-ar -ru
ID=/usr/bin/id -u

