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

set(INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${DYNAMIC_LIB_DIR})
set(LICENSE_DIR ${CPP_PLATFORM_HOME}/../licenses)

add_custom_target(licenses ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CPP_PLATFORM_HOME}
  COMMAND ${CMAKE_COMMAND} -E copy_directory licenses ${LICENSE_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(
    third_party
    DEPENDS eigen 3rd_party.sh
    COMMAND bash -c "./3rd_party.sh --add ${INSTALL_DIR}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Pull the Eigen repo as part of the configuration step
# thus avoiding any race conditions with parallel builds
execute_process(
    COMMAND ${CMAKE_COMMAND} -P "./pull-eigen.cmake"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_SOURCE_DIR}/3rd_party/eigen)
