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

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/clone_git_dependency.cmake)

if ( NOT EXISTS valijson )
  ml_clone_git_dependency(
    NAME Valijson
    URL https://github.com/tristanpenman/valijson.git
    BRANCH v1.0.2
    DESTINATION valijson
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    )
endif()
