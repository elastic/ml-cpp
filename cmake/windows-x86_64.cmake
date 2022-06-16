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

message(STATUS "In windows-x86_64.cmake")

set(CPP_PLATFORM_HOME $ENV{CPP_SRC_HOME}/build/distribution/platform/windows-x86_64)

# this must be first
include ("${CMAKE_CURRENT_LIST_DIR}/functions.cmake")

# set the os variables for windows
include ("${CMAKE_CURRENT_LIST_DIR}/os/windows.cmake")

# set the architecture bits
include ("${CMAKE_CURRENT_LIST_DIR}/architecture/x86_64.cmake")

include ("${CMAKE_CURRENT_LIST_DIR}/compiler/vs2019.cmake")

message(STATUS "windows-x86_64: ML_COMPILE_DEFINITIONS = ${ML_COMPILE_DEFINITIONS}")
message(STATUS "windows-x86_64: ML_LIBRARY_PREFIX ${ML_LIBRARY_PREFIX}")
