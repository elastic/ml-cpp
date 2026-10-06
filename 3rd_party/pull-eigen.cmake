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
  # The GitLab host that serves Eigen is prone to transient "unable to handle
  # this request due to load" failures. A single failed clone used to take out
  # the whole build, so retry a few times with a short, increasing backoff
  # before giving up. Each attempt starts from a clean slate because a failed
  # clone can leave a partial directory behind.
  set(EIGEN_CLONE_MAX_ATTEMPTS 5)
  set(EIGEN_CLONE_BACKOFF_SECONDS 5)
  set(GIT_RESULT 1)
  foreach(attempt RANGE 1 ${EIGEN_CLONE_MAX_ATTEMPTS})
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E rm -rf eigen
      )
    execute_process(
      COMMAND git -c advice.detachedHead=false clone --depth=1 --branch=3.4.0 https://gitlab.com/libeigen/eigen.git
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
      RESULT_VARIABLE GIT_RESULT
      )
    if(GIT_RESULT EQUAL 0)
      break()
    endif()
    if(attempt LESS ${EIGEN_CLONE_MAX_ATTEMPTS})
      math(EXPR backoff "${attempt} * ${EIGEN_CLONE_BACKOFF_SECONDS}")
      message(WARNING "Failed to clone Eigen (attempt ${attempt}/${EIGEN_CLONE_MAX_ATTEMPTS}): git exited with ${GIT_RESULT}. Retrying in ${backoff}s.")
      execute_process(COMMAND ${CMAKE_COMMAND} -E sleep ${backoff})
    endif()
  endforeach()
  if(NOT GIT_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to clone Eigen from https://gitlab.com/libeigen/eigen.git after ${EIGEN_CLONE_MAX_ATTEMPTS} attempts: git exited with ${GIT_RESULT}. Check network connectivity, proxy settings, and git availability.")
  endif()
endif()
