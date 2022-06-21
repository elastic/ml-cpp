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

# Script to get the appropriate version of Eigen, if not already present.
#
# If updating this script ensure the license information is correct in the
# licenses sub-directory.

# This cmake script is expected to be called from a target or custom command with WORKING_DIRECTORY set to this file's location

# This is the file where Eigen stores its version
set(VERSION_FILE "eigen/Eigen/src/Core/util/Macros.h")

if(EXISTS ${VERSION_FILE})

  file(READ ${VERSION_FILE} TMPTXT)

  # We want Eigen version 3.4.0 for our current branch
  string(FIND "${TMPTXT}" "#define EIGEN_WORLD_VERSION 3" world)
  string(FIND "${TMPTXT}" "#define EIGEN_MAJOR_VERSION 4" major)
  string(FIND "${TMPTXT}" "#define EIGEN_MINOR_VERSION 0" minor)

  if(${world} EQUAL -1 OR ${major} EQUAL -1 OR ${minor} EQUAL -1)
    set(PULL_EIGEN TRUE)
  endif ()
else()
  set(PULL_EIGEN TRUE)
endif()

if(PULL_EIGEN)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E rm -rf eigen
    )
  execute_process(
    COMMAND git -c advice.detachedHead=false clone --depth=1 --branch=3.4.0 https://gitlab.com/libeigen/eigen.git
    )
endif()
