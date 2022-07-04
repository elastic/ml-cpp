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

project("ML Ver")

# Prepare values to substitute
set(PRODUCT_VERSION "${ML_VERSION_NUM}")
if("$ENV{VERSION_QUALIFIER}")
  set(PRODUCT_VERSION "${PRODUCT_VERSION}-$ENV{VERSION_QUALIFIER}")
endif()
if(NOT "$ENV{SNAPSHOT}" STREQUAL "no")
  set(PRODUCT_VERSION "${PRODUCT_VERSION}-SNAPSHOT")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # On Windows ignore file mode differences, as file modes don't really exist on Windows
  execute_process(COMMAND git -c core.fileMode=false update-index -q --refresh ERROR_FILE /dev/null OUTPUT_FILE /dev/null)
  execute_process(COMMAND git -c core.fileMode=false diff-index HEAD -- OUTPUT_VARIABLE UNCOMMITTED_CHANGES)
else()
  execute_process(COMMAND git update-index -q --refresh ERROR_FILE /dev/null OUTPUT_FILE /dev/null)
  execute_process(COMMAND git diff-index HEAD -- OUTPUT_VARIABLE UNCOMMITTED_CHANGES)
endif()

# Decide which build to perform based on whether there are uncommitted changes
message(STATUS "UNCOMMITTED_CHANGES ${UNCOMMITTED_CHANGES}")
if(UNCOMMITTED_CHANGES)
  message(STATUS "DEV_BUILD")
  message(STATUS "PRODUCT_VERSION ${PRODUCT_VERSION}")
  message(STATUS "ML_USER ${ML_USER}")
  message(STATUS "BUILD_YEAR ${BUILD_YEAR}")
  add_compile_definitions(DEV_BUILD PRODUCT_VERSION=${PRODUCT_VERSION} ML_USER=${ML_USER} BUILD_YEAR=${BUILD_YEAR})
else()
  message(STATUS "PRODUCT_VERSION ${PRODUCT_VERSION}")
  message(STATUS "ML_BUILD_STR ${ML_BUILD_STR}")
  message(STATUS "BUILD_YEAR ${BUILD_YEAR}")
  add_compile_definitions(PRODUCT_VERSION=${PRODUCT_VERSION} ML_BUILD_STR=${ML_BUILD_STR} BUILD_YEAR=${BUILD_YEAR})
endif()

set(SRCS
  CBuildInfo.cc
  )

ml_add_library(MlVer STATIC ${SRCS})
