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

# Script to get the latest version of Valijson, if not already present.
#
# Valijson must only be used in test code, _not_ any form of redistributable code.

# This cmake script is expected to be called from a target or custom command with WORKING_DIRECTORY set to this file's location

if ( NOT EXISTS valijson )
  # Retry the clone a few times with a short, increasing backoff to ride out
  # transient git hosting outages rather than failing the whole build on a
  # single blip. Each attempt starts clean because a failed clone can leave a
  # partial directory behind.
  set(VALIJSON_CLONE_MAX_ATTEMPTS 5)
  set(VALIJSON_CLONE_BACKOFF_SECONDS 5)
  set(GIT_RESULT 1)
  foreach(attempt RANGE 1 ${VALIJSON_CLONE_MAX_ATTEMPTS})
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E rm -rf valijson
      )
    execute_process(
      COMMAND git -c advice.detachedHead=false clone --depth=1 --branch=v1.0.2 https://github.com/tristanpenman/valijson.git
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
      RESULT_VARIABLE GIT_RESULT
      )
    if(GIT_RESULT EQUAL 0)
      break()
    endif()
    if(attempt LESS ${VALIJSON_CLONE_MAX_ATTEMPTS})
      math(EXPR backoff "${attempt} * ${VALIJSON_CLONE_BACKOFF_SECONDS}")
      message(WARNING "Failed to clone Valijson (attempt ${attempt}/${VALIJSON_CLONE_MAX_ATTEMPTS}): git exited with ${GIT_RESULT}. Retrying in ${backoff}s.")
      execute_process(COMMAND ${CMAKE_COMMAND} -E sleep ${backoff})
    endif()
  endforeach()
  if(NOT GIT_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to clone Valijson from https://github.com/tristanpenman/valijson.git after ${VALIJSON_CLONE_MAX_ATTEMPTS} attempts: git exited with ${GIT_RESULT}. Check network connectivity, proxy settings, and git availability.")
  endif()
endif()
