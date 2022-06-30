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

set(CPP_PLATFORM_HOME $ENV{CPP_SRC_HOME}/build/distribution/platform/darwin-aarch64)

if(DEFINED ENV{CPP_CROSS_COMPILE} AND "$ENV{CPP_CROSS_COMPILE}" STREQUAL "macosx")
  # the name of the target operating system
  set(CMAKE_SYSTEM_NAME Darwin)

  # Apple silicon requires at least Big Sur which is Darwin 20.1.
  # Note that we don't currently support cross compiling to macOS ARM architecture,
  # so this branch is tecnically redundant.
  set(CMAKE_SYSTEM_VERSION 20.1.0)

  set(CROSS_TARGET_PLATFORM  x86_64-apple-macosx10.14)
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

# include darwin specific settings
include("${CMAKE_CURRENT_LIST_DIR}/os/darwin.cmake")

# set the architecture bits
include("${CMAKE_CURRENT_LIST_DIR}/architecture/aarch64.cmake")

# include clang specific settings
include("${CMAKE_CURRENT_LIST_DIR}/compiler/clang.cmake")
##########################

