#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#

set(CPP_PLATFORM_HOME $ENV{CPP_SRC_HOME}/build/distribution/platform/linux-aarch64)

if(DEFINED ENV{CPP_CROSS_COMPILE} AND "$ENV{CPP_CROSS_COMPILE}" STREQUAL "aarch64")
  # the name of the target operating system
  set(CMAKE_SYSTEM_NAME Linux)

  # This is the system version of CentOS 7, which is what we currently cross compile for Linux on.
  # Should be incremented if the cross compile Docker image for linux-aarch64 is rebuilt.
  set(CMAKE_SYSTEM_VERSION 3.10.0-514.6.1.el7.x86_64)

  set(CROSS_TARGET_PLATFORM  aarch64-linux-gnu)
  set(CMAKE_SYSROOT  /usr/local/sysroot-${CROSS_TARGET_PLATFORM})

  # adjust the default behavior of the FIND_XXX() commands:
  # search programs in the host environment
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

  # search headers and libraries in the target environment
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
endif()

message(STATUS "CMAKE_SYSTEM_NAME ${CMAKE_SYSTEM_NAME}")


##########################
# this must be first
include("${CMAKE_CURRENT_LIST_DIR}/functions.cmake")

# include linux specific settings
include("${CMAKE_CURRENT_LIST_DIR}/os/linux.cmake")

# set the architecture bits
include("${CMAKE_CURRENT_LIST_DIR}/architecture/aarch64.cmake")

# include gcc specific settings
include("${CMAKE_CURRENT_LIST_DIR}/compiler/gcc.cmake")
##########################

