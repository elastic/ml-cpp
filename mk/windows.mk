#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

OS=Windows

CPP_PLATFORM_HOME=$(CPP_DISTRIBUTION_HOME)/platform/windows-x86_64

CC=cl
CXX=cl

# Generally we'll want to build with a DLL version of the C runtime library, but
# occasionally we may need to override this
ifndef CRT_OPT
CRT_OPT=-MD
endif

ifndef ML_DEBUG
OPTCFLAGS=-O2 -Qfast_transcendentals -Qvec-report:1
OPTCPPFLAGS=-DNDEBUG -DEXCLUDE_TRACE_LOGGING
endif

SHELL:=$(LOCAL_DRIVE):/PROGRA~1/Git/bin/bash.exe
# On 64 bit Windows Visual Studio 2017 is in C:\Program Files (x86) aka C:\PROGRA~2

# compiler and sdk are dependent on your local install, tweak them with overwriting VCBASE and WINSDKBASE in your .bashrc
# do not override them here
# default: VS Professional 2017
VCBASE?=$(shell cd /$(LOCAL_DRIVE) && cygpath -m -s "Program Files (x86)/Microsoft Visual Studio/2017/Professional")
WINSDKBASE?=$(shell cd /$(LOCAL_DRIVE) && cygpath -m -s "Program Files (x86)/Windows Kits")

# example compiler defaults for VS Build Tools 2017, c&p into your .bashrc, note 8.3 paths might be different on your install
# export VCBASE=PROGRA~2/MICROS~2/2017/BUILDT~1
# export WINSDKBASE=PROGRA~2/WI3CF2~1

VCVER:=$(shell ls -1 /$(LOCAL_DRIVE)/$(VCBASE)/VC/Tools/MSVC | tail -1)
UCRTVER:=$(shell ls -1 /$(LOCAL_DRIVE)/$(WINSDKBASE)/10/Include | tail -1)
VCINCLUDES=-I$(LOCAL_DRIVE):/$(VCBASE)/VC/Tools/MSVC/$(VCVER)/include
VCLDFLAGS=-LIBPATH:$(LOCAL_DRIVE):/$(VCBASE)/VC/Tools/MSVC/$(VCVER)/lib/x64
WINSDKINCLUDES=-I$(LOCAL_DRIVE):/$(WINSDKBASE)/10/Include/$(UCRTVER)/ucrt -I$(LOCAL_DRIVE):/$(WINSDKBASE)/8.0/Include/shared -I$(LOCAL_DRIVE):/$(WINSDKBASE)/8.0/Include/um
WINSDKLDFLAGS=-LIBPATH:$(LOCAL_DRIVE):/$(WINSDKBASE)/10/Lib/$(UCRTVER)/ucrt/x64 -LIBPATH:$(LOCAL_DRIVE):/$(WINSDKBASE)/8.0/Lib/win8/um/x64
CFLAGS=-nologo $(OPTCFLAGS) -W4 $(CRT_OPT) -EHsc -Zi -Gw -FS -Zc:inline -diagnostics:caret -utf-8
CXXFLAGS=-TP $(CFLAGS) -Zc:rvalueCast -Zc:strictStrings -wd4127 -we4150 -wd4201 -wd4231 -wd4251 -wd4355 -wd4512 -wd4702 -bigobj
ANALYZEFLAGS=-nologo -analyze:only -analyze:stacksize100000 $(CRT_OPT)

CPPFLAGS=-X -I$(CPP_SRC_HOME)/3rd_party/include -I$(LOCAL_DRIVE):/usr/local/include $(VCINCLUDES) $(WINSDKINCLUDES) -D$(OS) -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE -DWIN32_LEAN_AND_MEAN -DNTDDI_VERSION=0x06010000 -D_WIN32_WINNT=0x0601 -DBUILDING_$(basename $(notdir $(TARGET))) $(OPTCPPFLAGS)
# -MD defines _DLL and _MT - for dependency determination we must define these
# otherwise the Boost headers will throw errors during preprocessing
ifeq ($(CRT_OPT),-MD)
CDEPFLAGS=-nologo -E -D_DLL -D_MT
else
CDEPFLAGS=-nologo -E -D_MT
endif
COMP_OUT_FLAG=-Fo
ANALYZE_OUT_FLAG=-analyze:log
LINK_OUT_FLAG=-Fe
AR_OUT_FLAG=-OUT:
# Get the dependencies that aren't under C:\usr\local or C:\Program Files*, on
# the assumption that the 3rd party tools won't change very often, and if they
# do then we'll rebuild everything from scratch
DEP_FILTER= 2>/dev/null | egrep "^.line .*(\\.h|$<)" | tr -s '\\\\' '/' | awk -F'"' '{ print $$2 }' | egrep -i -v "usr.local|$(LOCAL_DRIVE)..progra" | sed 's~/[a-z]*/\.\./~/~g' | sort -f -u | sort -t. -k2 | tr '\r\n\t' ' ' | sed 's/  / /g' | sed 's/^ //' | sed 's/ $$//'
DEP_REFORMAT=sed 's,$<,$(basename $@)$(OBJECT_FILE_EXT) $@ : $<,'
OBJECT_FILE_EXT=.obj
EXE_EXT=.exe
EXE_DIR=bin
DYNAMIC_LIB_EXT=.dll
DYNAMIC_LIB_DIR=bin
IMPORT_LIB_DIR=lib
RESOURCE_FILE=$(OBJS_DIR)/ml.res
STATIC_LIB_EXT=.lib
SHELL_SCRIPT_EXT=.bat
# This temp directory assumes we're running in a Unix-like shell such as Git bash
UT_TMP_DIR=/tmp
RESOURCES_DIR=resources
LOCALLIBS=AdvAPI32.lib shell32.lib Version.lib
NETLIBS=WS2_32.lib
BOOSTVER=1_71
BOOSTVCVER=141
BOOSTINCLUDES=-I$(LOCAL_DRIVE):/usr/local/include/boost-$(BOOSTVER)
BOOSTCPPFLAGS=-DBOOST_ALL_DYN_LINK -DBOOST_ALL_NO_LIB -DBOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
BOOSTLOGLIBS=boost_log-vc$(BOOSTVCVER)-mt-x64-$(BOOSTVER).lib
BOOSTLOGSETUPLIBS=boost_log_setup-vc$(BOOSTVCVER)-mt-x64-$(BOOSTVER).lib
BOOSTREGEXLIBS=boost_regex-vc$(BOOSTVCVER)-mt-x64-$(BOOSTVER).lib
BOOSTIOSTREAMSLIBS=boost_iostreams-vc$(BOOSTVCVER)-mt-x64-$(BOOSTVER).lib
BOOSTPROGRAMOPTIONSLIBS=boost_program_options-vc$(BOOSTVCVER)-mt-x64-$(BOOSTVER).lib
BOOSTTHREADLIBS=boost_thread-vc$(BOOSTVCVER)-mt-x64-$(BOOSTVER).lib boost_chrono-vc$(BOOSTVCVER)-mt-x64-$(BOOSTVER).lib boost_system-vc$(BOOSTVCVER)-mt-x64-$(BOOSTVER).lib
BOOSTFILESYSTEMLIBS=boost_filesystem-vc$(BOOSTVCVER)-mt-x64-$(BOOSTVER).lib boost_system-vc$(BOOSTVCVER)-mt-x64-$(BOOSTVER).lib
BOOSTDATETIMELIBS=boost_date_time-vc$(BOOSTVCVER)-mt-x64-$(BOOSTVER).lib
BOOSTTESTLIBS=boost_unit_test_framework-vc$(BOOSTVCVER)-mt-x64-$(BOOSTVER).lib
RAPIDJSONINCLUDES=-I$(CPP_SRC_HOME)/3rd_party/rapidjson/include
RAPIDJSONCPPFLAGS=-DRAPIDJSON_HAS_STDSTRING -DRAPIDJSON_SSE42
# Eigen automatically uses SSE and SSE2 on 64 bit Windows - only the higher
# versions need to be explicitly enabled
EIGENINCLUDES=-isystem $(CPP_SRC_HOME)/3rd_party/eigen
EIGENCPPFLAGS=-DEIGEN_MPL2_ONLY -DEIGEN_VECTORIZE_SSE3 -DEIGEN_VECTORIZE_SSE4_1 -DEIGEN_VECTORIZE_SSE4_2
XMLINCLUDES=-I$(LOCAL_DRIVE):/usr/local/include/libxml2
XMLLIBLDFLAGS=-LIBPATH:$(LOCAL_DRIVE):/usr/local/lib
XMLLIBS=libxml2.lib
DYNAMICLIBLDFLAGS=-nologo -Zi $(CRT_OPT) -LD -link -MAP -OPT:REF -INCREMENTAL:NO -LIBPATH:$(CPP_PLATFORM_HOME)/$(IMPORT_LIB_DIR)
ZLIBLIBS=zdll.lib
STRPTIMELIBS=strptime.lib
EXELDFLAGS=-nologo -Zi $(CRT_OPT) -link -MAP -OPT:REF -SUBSYSTEM:CONSOLE,6.1 -STACK:0x800000 -INCREMENTAL:NO -LIBPATH:$(CPP_PLATFORM_HOME)/$(IMPORT_LIB_DIR)
UTLDFLAGS=$(EXELDFLAGS)
INSTALL=cp
CP=cp
RC=rc -nologo
LIB_ML_CORE=libMlCore.lib
LIB_ML_VER=libMlVer.lib
ML_VER_LDFLAGS=-LIBPATH:$(CPP_SRC_HOME)/lib/ver/.objs
LIB_ML_API=libMlApi.lib
LIB_ML_MATHS=libMlMaths.lib
LIB_ML_CONFIG=libMlConfig.lib
LIB_ML_MODEL=libMlModel.lib
LIB_ML_SECCOMP=libMlSeccomp.lib
ML_SECCOMP_LDFLAGS=-LIBPATH:$(CPP_SRC_HOME)/lib/seccomp/.objs
LIB_ML_TEST=libMlTest.lib

LIB_PATH+=-LIBPATH:$(LOCAL_DRIVE):/usr/local/lib $(VCLDFLAGS) $(WINSDKLDFLAGS)
PDB_FLAGS=-Fd$(basename $(TARGET)).pdb

MKDIR=mkdir -p
RM=rm -f
RMDIR=rm -rf
MV=mv -f
SED=sed
ECHO=echo
CAT=cat
LN=ln
AR=lib -NOLOGO
ID=id -u

