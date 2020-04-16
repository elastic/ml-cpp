#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
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
ifeq ($(CPP_CROSS_COMPILE),macosx)
include $(CPP_SRC_HOME)/mk/linux_crosscompile_macosx.mk
else
include $(CPP_SRC_HOME)/mk/linux_crosscompile_linux.mk
endif
else
include $(CPP_SRC_HOME)/mk/linux.mk
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

