#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

OS=MacOSX

HARDWARE_ARCH:=$(shell uname -m | sed 's/arm64/aarch64/')
CPP_PLATFORM_HOME=$(CPP_DISTRIBUTION_HOME)/platform/darwin-$(HARDWARE_ARCH)
ML_APP_NAME=controller
APP_CONTENTS=$(ML_APP_NAME).app/Contents

CC=clang
CXX=clang++ -std=c++17 -stdlib=libc++

ifndef ML_DEBUG
OPTCFLAGS=-O3
OPTCPPFLAGS=-DNDEBUG -DEXCLUDE_TRACE_LOGGING
endif

ifdef ML_DEBUG
ifdef ML_COVERAGE
COVERAGE=--coverage
endif
endif

ifeq ($(HARDWARE_ARCH),x86_64)
ARCHCFLAGS=-msse4.2
endif

SDK_PATH:=$(shell xcrun --show-sdk-path)
# Start by enabling all warnings and then disable the really pointless/annoying ones
CFLAGS=-g $(OPTCFLAGS) $(ARCHCFLAGS) -fstack-protector -Weverything -Werror-switch -Wno-deprecated -Wno-disabled-macro-expansion -Wno-documentation-deprecated-sync -Wno-documentation-unknown-command -Wno-float-equal -Wno-missing-prototypes -Wno-padded -Wno-poison-system-directories -Wno-sign-conversion -Wno-unreachable-code -Wno-used-but-marked-unused $(COVERAGE)
CXXFLAGS=$(CFLAGS) -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-exit-time-destructors -Wno-global-constructors -Wno-return-std-move-in-c++11 -Wno-unused-member-function -Wno-weak-vtables
CPPFLAGS=-isystem $(CPP_SRC_HOME)/3rd_party/include -isystem /usr/local/include -D$(OS) $(OPTCPPFLAGS)
ANALYZEFLAGS=--analyze
CDEPFLAGS=-MM
COMP_OUT_FLAG=-o
ANALYZE_OUT_FLAG=-o
LINK_OUT_FLAG=-o
DEP_REFORMAT=sed 's,\($*\)\.o[ :]*,$(OBJS_DIR)\/\1.o $@ : ,g'
OBJECT_FILE_EXT=.o
EXE_DIR=$(APP_CONTENTS)/MacOS
DYNAMIC_LIB_EXT=.dylib
DYNAMIC_LIB_DIR=$(APP_CONTENTS)/lib
STATIC_LIB_EXT=.a
SHELL_SCRIPT_EXT=.sh
UT_TMP_DIR=/tmp/$(LOGNAME)
RESOURCES_DIR=$(APP_CONTENTS)/Resources
LOCALLIBS=
NETLIBS=
BOOSTVER=1_71
ifeq ($(HARDWARE_ARCH),x86_64)
BOOSTARCH=x64
else
BOOSTARCH=a64
endif
BOOSTCLANGVER:=$(shell $(CXX) --version | grep ' version ' | sed 's/.* version //' | awk -F. '{ print $$1$$2; }')
# Use -isystem instead of -I for Boost headers to suppress warnings from Boost
BOOSTINCLUDES=-isystem /usr/local/include/boost-$(BOOSTVER)
BOOSTCPPFLAGS=-DBOOST_ALL_DYN_LINK -DBOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
BOOSTLOGLIBS=-lboost_log-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
BOOSTLOGSETUPLIBS=-lboost_log_setup-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
BOOSTREGEXLIBS=-lboost_regex-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
BOOSTIOSTREAMSLIBS=-lboost_iostreams-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
BOOSTPROGRAMOPTIONSLIBS=-lboost_program_options-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
BOOSTTHREADLIBS=-lboost_thread-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTARCH)-$(BOOSTVER) -lboost_system-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
BOOSTFILESYSTEMLIBS=-lboost_filesystem-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTARCH)-$(BOOSTVER) -lboost_system-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
BOOSTDATETIMELIBS=-lboost_date_time-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
BOOSTTESTLIBS=-lboost_unit_test_framework-clang-darwin$(BOOSTCLANGVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
RAPIDJSONINCLUDES=-isystem $(CPP_SRC_HOME)/3rd_party/rapidjson/include
ifeq ($(HARDWARE_ARCH),x86_64)
RAPIDJSONCPPFLAGS=-DRAPIDJSON_HAS_STDSTRING -DRAPIDJSON_SSE42
else
RAPIDJSONCPPFLAGS=-DRAPIDJSON_HAS_STDSTRING -DRAPIDJSON_NEON
endif
EIGENINCLUDES=-isystem $(CPP_SRC_HOME)/3rd_party/eigen
EIGENCPPFLAGS=-DEIGEN_MPL2_ONLY -DEIGEN_MAX_ALIGN_BYTES=32
TORCHINCLUDES=-isystem /usr/local/include/pytorch
TORCHCPULIB=-ltorch_cpu
C10LIB=-lc10
XMLINCLUDES=-isystem $(SDK_PATH)/usr/include/libxml2
XMLLIBLDFLAGS=-L/usr/lib
XMLLIBS=-lxml2
JAVANATIVEINCLUDES=-I`/usr/libexec/java_home`/include
JAVANATIVELDFLAGS=-L`/usr/libexec/java_home`/jre/lib/server
JAVANATIVELIBS=-ljvm
ML_VERSION_NUM:=$(shell cat $(CPP_SRC_HOME)/gradle.properties | grep '^elasticsearchVersion' | awk -F= '{ print $$2 }' | xargs echo | sed 's/-.*//')
DYNAMICLIBLDFLAGS=-current_version $(ML_VERSION_NUM) -compatibility_version $(ML_VERSION_NUM) -dynamiclib -Wl,-dead_strip_dylibs $(COVERAGE) -Wl,-install_name,@rpath/$(notdir $(TARGET)) -L$(CPP_PLATFORM_HOME)/$(DYNAMIC_LIB_DIR) -Wl,-rpath,@loader_path/. -Wl,-headerpad_max_install_names
ZLIBLIBS=-lz
EXELDFLAGS=-bind_at_load -L$(CPP_PLATFORM_HOME)/$(DYNAMIC_LIB_DIR) $(COVERAGE) -Wl,-rpath,@loader_path/../lib -Wl,-headerpad_max_install_names
UTLDFLAGS=$(EXELDFLAGS) -Wl,-rpath,$(CPP_PLATFORM_HOME)/$(DYNAMIC_LIB_DIR)
PLIST_FILE=$(OBJS_DIR)/Info.plist
PLIST_FILE_LDFLAGS=-Wl,-sectcreate,__TEXT,__info_plist,$(PLIST_FILE)
LIB_ML_API=-lMlApi
LIB_ML_CORE=-lMlCore
LIB_ML_VER=-lMlVer
ML_VER_LDFLAGS=-L$(CPP_SRC_HOME)/lib/ver/.objs
LIB_ML_MATHS=-lMlMaths
LIB_ML_MODEL=-lMlModel
LIB_ML_SECCOMP=-lMlSeccomp
ML_SECCOMP_LDFLAGS=-L$(CPP_SRC_HOME)/lib/seccomp/.objs
LIB_ML_TEST=-lMlTest

LIB_PATH+=-L/usr/local/lib

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
AR=ar -ru
ID=/usr/bin/id -u

