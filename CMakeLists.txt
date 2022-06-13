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

# CMake 3.19.2 is the minimum version required to support Apple Silicon
cmake_minimum_required(VERSION 3.19.2)

project("ML")

set (CMAKE_BASE_ROOT_DIR ${CMAKE_SOURCE_DIR})
list (FIND CMAKE_MODULE_PATH "${CMAKE_BASE_ROOT_DIR}/cmake" _index)
if (${_index} EQUAL -1)
  list (APPEND CMAKE_MODULE_PATH "${CMAKE_BASE_ROOT_DIR}/cmake")
endif()

message (STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
message (STATUS "CMAKE_VERSION: ${CMAKE_VERSION}")

include(variables)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  add_custom_target(
    plist ALL
    DEPENDS $ENV{CPP_SRC_HOME}/mk/make_info_plist.sh
    DEPENDS $ENV{CPP_SRC_HOME}/gradle.properties
    COMMAND $ENV{CPP_SRC_HOME}/mk/make_info_plist.sh "${ML_APP_NAME}" false > Info.plist
    )
  install(FILES ${CMAKE_BINARY_DIR}/Info.plist DESTINATION ${CMAKE_INSTALL_PREFIX})
endif()

message(STATUS "CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}")

include_directories(SYSTEM ${ML_SYSTEM_INCLUDE_DIRECTORIES})

add_compile_definitions(${ML_COMPILE_DEFINITIONS})
add_compile_options(${ML_CXX_FLAGS})


add_subdirectory(3rd_party)
add_subdirectory(lib)
add_subdirectory(bin)
add_subdirectory(test)

# Add a target to build Doxygen generated documentation
# if the doxygen executable can be found
include(doxygen)
ml_doxygen(${CMAKE_SOURCE_DIR}/build/doxygen)
