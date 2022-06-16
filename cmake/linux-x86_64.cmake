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

# the name of the target operating system
set(CMAKE_SYSTEM_NAME Linux)

set(CPP_PLATFORM_HOME $ENV{CPP_SRC_HOME}/build/distribution/platform/linux-x86_64)

message(STATUS "CMAKE_SYSTEM_NAME ${CMAKE_SYSTEM_NAME}")

# this must be first
include("${CMAKE_CURRENT_LIST_DIR}/functions.cmake")

# include linux specific settings
include("${CMAKE_CURRENT_LIST_DIR}/os/linux.cmake")

# set the architecture bits
include("${CMAKE_CURRENT_LIST_DIR}/architecture/x86_64.cmake")

# include gcc specific settings
include("${CMAKE_CURRENT_LIST_DIR}/compiler/gcc.cmake")
