#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

OS=Linux

HARDWARE_ARCH=$(CPP_CROSS_COMPILE)
CPP_PLATFORM_HOME=$(CPP_DISTRIBUTION_HOME)/platform/linux-$(HARDWARE_ARCH)

CROSS_TARGET_PLATFORM=$(CPP_CROSS_COMPILE)-linux-gnu
SYSROOT=/usr/local/sysroot-$(CROSS_TARGET_PLATFORM)
CROSS_FLAGS=--sysroot=$(SYSROOT)
CC=$(CROSS_TARGET_PLATFORM)-gcc $(CROSS_FLAGS)
CXX=$(CROSS_TARGET_PLATFORM)-g++ $(CROSS_FLAGS) -std=gnu++17

ifndef ML_DEBUG
OPTCFLAGS=-O3 -Wdisabled-optimization
# Fortify Source can only be used with optimisation
OPTCPPFLAGS=-DNDEBUG -DEXCLUDE_TRACE_LOGGING -D_FORTIFY_SOURCE=2
endif

ifdef ML_DEBUG
ifdef ML_COVERAGE
COVERAGE=--coverage
endif
endif

ifeq ($(HARDWARE_ARCH),aarch64)
ARCHCFLAGS=-march=armv8-a+crc+crypto
endif

PLATPICFLAGS=-fPIC
PLATPIEFLAGS=-fPIE
CFLAGS=-g $(OPTCFLAGS) $(ARCHCFLAGS) -fstack-protector -fno-math-errno -Wall -Wcast-align -Wconversion -Wextra -Winit-self -Wparentheses -Wpointer-arith -Wswitch-enum $(COVERAGE)
CXXFLAGS=$(CFLAGS) -Wno-ctor-dtor-privacy -Wno-deprecated-declarations -Wold-style-cast -fvisibility-inlines-hidden
CPPFLAGS=-isystem $(CPP_SRC_HOME)/3rd_party/include -isystem $(SYSROOT)/usr/local/gcc93/include -D$(OS) -D_REENTRANT $(OPTCPPFLAGS)
CDEPFLAGS=-MM
COMP_OUT_FLAG=-o
LINK_OUT_FLAG=-o
DEP_REFORMAT=sed 's,\($*\)\.o[ :]*,$(OBJS_DIR)\/\1.o $@ : ,g'
OBJECT_FILE_EXT=.o
EXE_DIR=bin
DYNAMIC_LIB_EXT=.so
DYNAMIC_LIB_DIR=lib
STATIC_LIB_EXT=.a
SHELL_SCRIPT_EXT=.sh
UT_TMP_DIR=/tmp/$(LOGNAME)
RESOURCES_DIR=resources
LOCALLIBS=-lm -lpthread -ldl -lrt
NETLIBS=-lnsl
BOOSTVER=1_71
ifeq ($(HARDWARE_ARCH),aarch64)
BOOSTARCH=a64
else
BOOSTARCH=not_supported
endif
BOOSTGCCVER:=$(shell $(CXX) -dumpversion | awk -F. '{ print $$1; }')
# Use -isystem instead of -I for Boost headers to suppress warnings from Boost
BOOSTINCLUDES=-isystem $(SYSROOT)/usr/local/gcc93/include/boost-$(BOOSTVER)
BOOSTCPPFLAGS=-DBOOST_ALL_DYN_LINK -DBOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
BOOSTLOGLIBS=-lboost_log-gcc$(BOOSTGCCVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
BOOSTLOGSETUPLIBS=-lboost_log_setup-gcc$(BOOSTGCCVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
BOOSTREGEXLIBS=-lboost_regex-gcc$(BOOSTGCCVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
BOOSTIOSTREAMSLIBS=-lboost_iostreams-gcc$(BOOSTGCCVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
BOOSTPROGRAMOPTIONSLIBS=-lboost_program_options-gcc$(BOOSTGCCVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
BOOSTTHREADLIBS=-lboost_thread-gcc$(BOOSTGCCVER)-mt-$(BOOSTARCH)-$(BOOSTVER) -lboost_system-gcc$(BOOSTGCCVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
BOOSTFILESYSTEMLIBS=-lboost_filesystem-gcc$(BOOSTGCCVER)-mt-$(BOOSTARCH)-$(BOOSTVER) -lboost_system-gcc$(BOOSTGCCVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
BOOSTDATETIMELIBS=-lboost_date_time-gcc$(BOOSTGCCVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
BOOSTTESTLIBS=-lboost_unit_test_framework-gcc$(BOOSTGCCVER)-mt-$(BOOSTARCH)-$(BOOSTVER)
RAPIDJSONINCLUDES=-isystem $(CPP_SRC_HOME)/3rd_party/rapidjson/include
ifeq ($(HARDWARE_ARCH),aarch64)
RAPIDJSONCPPFLAGS=-DRAPIDJSON_HAS_STDSTRING -DRAPIDJSON_NEON
else
RAPIDJSONCPPFLAGS=-DRAPIDJSON_HAS_STDSTRING
endif
EIGENINCLUDES=-isystem $(CPP_SRC_HOME)/3rd_party/eigen
EIGENCPPFLAGS=-DEIGEN_MPL2_ONLY -DEIGEN_MAX_ALIGN_BYTES=32
TORCHINCLUDES=-isystem $(SYSROOT)/usr/local/gcc93/include/pytorch
TORCHCPULIB=-ltorch_cpu
C10LIB=-lc10
XMLINCLUDES=-I$(SYSROOT)/usr/local/gcc93/include/libxml2
XMLLIBS=-L$(SYSROOT)/usr/local/gcc93/lib -lxml2 -lz -lm -ldl
DYNAMICLIBLDFLAGS=$(PLATPICFLAGS) -shared -Wl,--as-needed -L$(CPP_PLATFORM_HOME)/$(DYNAMIC_LIB_DIR) $(COVERAGE) -Wl,-z,relro -Wl,-z,now -Wl,-rpath,'$$ORIGIN/.'
ZLIBLIBS=-lz
EXELDFLAGS=-pie $(PLATPIEFLAGS) -L$(CPP_PLATFORM_HOME)/$(DYNAMIC_LIB_DIR) $(COVERAGE) -Wl,-z,relro -Wl,-z,now -Wl,-rpath,'$$ORIGIN/../lib'
UTLDFLAGS=$(EXELDFLAGS) -Wl,-rpath,$(CPP_PLATFORM_HOME)/$(DYNAMIC_LIB_DIR)
LIB_ML_CORE=-lMlCore
LIB_ML_VER=-lMlVer
ML_VER_LDFLAGS=-L$(CPP_SRC_HOME)/lib/ver/.objs
LIB_ML_API=-lMlApi
LIB_ML_MATHS=-lMlMaths
LIB_ML_MODEL=-lMlModel
LIB_ML_SECCOMP=-lMlSeccomp
ML_SECCOMP_LDFLAGS=-L$(CPP_SRC_HOME)/lib/seccomp/.objs
LIB_ML_TEST=-lMlTest

LIB_PATH+=-L$(SYSROOT)/usr/local/gcc93/lib

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
AR=$(CROSS_TARGET_PLATFORM)-ar -rus
ID=/usr/bin/id -u

