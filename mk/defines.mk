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

# OS and makefile suffices
# Mac
macOS=Darwin

# Linux
linuxOS=Linux

# Windows
windows=MINGW

# Assume a snapshot build unless told otherwise
SNAPSHOT?=yes

# If building in one of these directories we're building within a dedicated
# container (Docker or Vagrant), and can skip certain checks like we would under
# Jenkins
ifeq ($(CPP_SRC_HOME),/ml-cpp)
JOB_NAME?=container-build
else
ifeq ($(CPP_SRC_HOME),C:/Users/vagrant/projects/elasticsearch-extra/ml-cpp)
JOB_NAME?=container-build
endif
endif

OS:=$(shell uname)
REV:=$(shell uname -r)

# Will return FQDN (Fully Qualified Domain Name) on all platforms except
# Windows (MinGW).  On Windows (MinGW) it will return the name of the machine.
HOSTNAME:=$(shell uname -n)

CPP_DISTRIBUTION_HOME=$(CPP_SRC_HOME)/build/distribution

# Detect Linux
ifeq ($(OS),$(linuxOS))
ifdef CPP_CROSS_COMPILE
include $(CPP_SRC_HOME)/mk/linux_crosscompile_$(CPP_CROSS_COMPILE).mk
else
MUSL:=$(shell ldd --version 2>&1 | grep musl)
ifeq ($(MUSL),)
include $(CPP_SRC_HOME)/mk/linux.mk
else
include $(CPP_SRC_HOME)/mk/linux-musl.mk
endif
endif
endif

# Detect MacOSX
ifeq ($(OS),$(macOS))
include $(CPP_SRC_HOME)/mk/macosx.mk
endif

# Detect Windows
ifeq ($(patsubst $(windows)%,$(windows),$(OS)),$(windows))
# If this default local drive letter is wrong, it can be overridden using an
# environment variable
LOCAL_DRIVE=C
include $(CPP_SRC_HOME)/mk/windows.mk
endif

OBJS_DIR=.objs

