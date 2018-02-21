#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

OS=Linux

CPP_PLATFORM_HOME=$(CPP_DISTRIBUTION_HOME)/platform/linux-x86_64

CC=gcc
CXX=g++ -std=gnu++0x

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

PLATPICFLAGS=-fPIC
PLATPIEFLAGS=-fPIE
CFLAGS=-g $(OPTCFLAGS) -msse4.2 -mfpmath=sse -fstack-protector -fno-math-errno -fno-permissive -Wall -Wcast-align -Wconversion -Wextra -Winit-self -Wparentheses -Wpointer-arith -Wswitch-enum $(COVERAGE)
CXXFLAGS=$(CFLAGS) -Wno-ctor-dtor-privacy -Wno-deprecated-declarations -Wold-style-cast -fvisibility-inlines-hidden
CPPFLAGS=-isystem $(CPP_SRC_HOME)/3rd_party/include -isystem /usr/local/gcc62/include -D$(OS) -DLINUX=2 -D_REENTRANT $(OPTCPPFLAGS)
CDEPFLAGS=-MM
COMP_OUT_FLAG=-o 
LINK_OUT_FLAG=-o 
DEP_REFORMAT=sed 's,\($*\)\.o[ :]*,$(OBJS_DIR)\/\1.o $@ : ,g'
LOCALLIBS=-lm -lpthread -ldl -lrt
NETLIBS=-lnsl
BOOSTVER=1_65_1
BOOSTGCCVER=$(shell $(CXX) -dumpversion | awk -F. '{ print $$1$$2; }')
# Use -isystem instead of -I for Boost headers to suppress warnings from Boost
BOOSTINCLUDES=-isystem /usr/local/gcc62/include/boost-$(BOOSTVER)
BOOSTCPPFLAGS=-DBOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
BOOSTREGEXLIBS=-lboost_regex-gcc$(BOOSTGCCVER)-mt-$(BOOSTVER)
BOOSTIOSTREAMSLIBS=-lboost_iostreams-gcc$(BOOSTGCCVER)-mt-$(BOOSTVER)
BOOSTPROGRAMOPTIONSLIBS=-lboost_program_options-gcc$(BOOSTGCCVER)-mt-$(BOOSTVER)
BOOSTTHREADLIBS=-lboost_thread-gcc$(BOOSTGCCVER)-mt-$(BOOSTVER) -lboost_system-gcc$(BOOSTGCCVER)-mt-$(BOOSTVER)
BOOSTFILESYSTEMLIBS=-lboost_filesystem-gcc$(BOOSTGCCVER)-mt-$(BOOSTVER) -lboost_system-gcc$(BOOSTGCCVER)-mt-$(BOOSTVER)
BOOSTDATETIMELIBS=-lboost_date_time-gcc$(BOOSTGCCVER)-mt-$(BOOSTVER)
RAPIDJSONINCLUDES=-isystem $(CPP_SRC_HOME)/3rd_party/rapidjson/include
RAPIDJSONCPPFLAGS=-DRAPIDJSON_HAS_STDSTRING -DRAPIDJSON_SSE42
EIGENCPPFLAGS=-DEIGEN_MPL2_ONLY
XMLINCLUDES=`/usr/local/gcc62/bin/xml2-config --cflags`
XMLLIBS=`/usr/local/gcc62/bin/xml2-config --libs`
DYNAMICLIBLDFLAGS=$(PLATPICFLAGS) -shared -Wl,--as-needed -L$(CPP_PLATFORM_HOME)/lib $(COVERAGE) -Wl,-z,relro -Wl,-z,now -Wl,-rpath,'$$ORIGIN/.'
JAVANATIVEINCLUDES=-I$(JAVA_HOME)/include
JAVANATIVELDFLAGS=-L$(JAVA_HOME)/jre/lib/server
JAVANATIVELIBS=-ljvm
CPPUNITLIBS=-lcppunit
LOG4CXXLIBS=-llog4cxx
ZLIBLIBS=-lz
EXELDFLAGS=-pie $(PLATPIEFLAGS) -L$(CPP_PLATFORM_HOME)/lib $(COVERAGE) -Wl,-z,relro -Wl,-z,now -Wl,-rpath,'$$ORIGIN/../lib'
UTLDFLAGS=$(EXELDFLAGS) -Wl,-rpath,$(CPP_PLATFORM_HOME)/lib
OBJECT_FILE_EXT=.o
DYNAMIC_LIB_EXT=.so
DYNAMIC_LIB_DIR=lib
STATIC_LIB_EXT=.a
SHELL_SCRIPT_EXT=.sh
UT_TMP_DIR=/tmp/$(LOGNAME)
LIB_ML_CORE=-lMlCore
LIB_ML_VER=-lMlVer
ML_VER_LDFLAGS=-L$(CPP_SRC_HOME)/lib/ver/.objs
LIB_ML_API=-lMlApi
LIB_ML_MATHS=-lMlMaths
LIB_ML_CONFIG=-lMlConfig
LIB_ML_MODEL=-lMlModel
LIB_ML_TEST=-lMlTest

LIB_PATH+=-L/usr/local/gcc62/lib

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
AR=ar -rus
ID=/usr/bin/id -u

