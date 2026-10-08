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

# Helper used by the 3rd_party/pull-*.cmake scripts to fetch header-only
# dependencies. It is kept in its own module (rather than cmake/functions.cmake)
# so that it can be include()d from `cmake -P` script-mode invocations without
# pulling in the project-configuration targets defined there.

#
# Clone a 3rd-party git dependency with a bounded retry loop.
#
# The hosts that serve our 3rd-party sources (gitlab.com, github.com) sometimes
# return transient errors under load, and a single failed clone used to take out
# an entire CI build. Retry a few times with a short, increasing backoff before
# giving up, starting from a clean slate on every attempt because a failed clone
# can leave a partial directory behind. A FATAL_ERROR is raised once the retries
# are exhausted so the caller (via COMMAND_ERROR_IS_FATAL) stops immediately with
# a clear message rather than failing later with a cryptic missing-header error.
#
# Named arguments:
#   NAME              human-readable dependency name used in log messages
#   URL               git repository URL to clone
#   BRANCH            branch or tag to check out (shallow, --depth=1)
#   DESTINATION       directory the repo is cloned into
#   WORKING_DIRECTORY directory in which the clone is performed
#   MAX_ATTEMPTS      optional number of attempts (default 5)
#   BACKOFF_SECONDS   optional base backoff, multiplied by the attempt number (default 5)
#
function(ml_clone_git_dependency)
  cmake_parse_arguments(CLONE "" "NAME;URL;BRANCH;DESTINATION;WORKING_DIRECTORY;MAX_ATTEMPTS;BACKOFF_SECONDS" "" ${ARGN})

  if(NOT CLONE_MAX_ATTEMPTS)
    set(CLONE_MAX_ATTEMPTS 5)
  endif()
  if(NOT CLONE_BACKOFF_SECONDS)
    set(CLONE_BACKOFF_SECONDS 5)
  endif()

  set(GIT_RESULT 1)
  foreach(attempt RANGE 1 ${CLONE_MAX_ATTEMPTS})
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E rm -rf ${CLONE_DESTINATION}
      WORKING_DIRECTORY ${CLONE_WORKING_DIRECTORY}
      )
    execute_process(
      COMMAND git -c advice.detachedHead=false clone --depth=1 --branch=${CLONE_BRANCH} ${CLONE_URL} ${CLONE_DESTINATION}
      WORKING_DIRECTORY ${CLONE_WORKING_DIRECTORY}
      RESULT_VARIABLE GIT_RESULT
      )
    if(GIT_RESULT EQUAL 0)
      break()
    endif()
    if(attempt LESS ${CLONE_MAX_ATTEMPTS})
      math(EXPR backoff "${attempt} * ${CLONE_BACKOFF_SECONDS}")
      message(WARNING "Failed to clone ${CLONE_NAME} (attempt ${attempt}/${CLONE_MAX_ATTEMPTS}): git exited with ${GIT_RESULT}. Retrying in ${backoff}s.")
      execute_process(COMMAND ${CMAKE_COMMAND} -E sleep ${backoff})
    endif()
  endforeach()

  if(NOT GIT_RESULT EQUAL 0)
    # Remove any partial checkout left by the final failed attempt so that a
    # subsequent configure re-attempts the clone instead of seeing a leftover
    # directory, skipping the clone, and failing much later with a cryptic
    # missing-header compile error.
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E rm -rf ${CLONE_DESTINATION}
      WORKING_DIRECTORY ${CLONE_WORKING_DIRECTORY}
      )
    message(FATAL_ERROR "Failed to clone ${CLONE_NAME} from ${CLONE_URL} after ${CLONE_MAX_ATTEMPTS} attempts: git exited with ${GIT_RESULT}. Check network connectivity, proxy settings, and git availability.")
  endif()
endfunction()
