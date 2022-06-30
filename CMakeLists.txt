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

set (CMAKE_BASE_ROOT_DIR ${CMAKE_SOURCE_DIR})
list (FIND CMAKE_MODULE_PATH "${CMAKE_BASE_ROOT_DIR}/cmake" _index)
if (${_index} EQUAL -1)
  list (APPEND CMAKE_MODULE_PATH "${CMAKE_BASE_ROOT_DIR}/cmake")
endif()

message (STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
message (STATUS "CMAKE_VERSION: ${CMAKE_VERSION}")

# Our CI build scripts explicitly specify the toolchain file
# on the command line. To simplify developer builds allow
# the toolchain to be obtained from an environment variable
# failing that the toolchain is determined from host processor
# and platform
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  if(DEFINED ENV{CMAKE_TOOLCHAIN_FILE})
    set(CMAKE_TOOLCHAIN_FILE $ENV{CMAKE_TOOLCHAIN_FILE})
  else()
    # Determine which toolchain to use based on host platform and architecture
    string(TOLOWER ${CMAKE_HOST_SYSTEM_NAME} HOST_SYSTEM_NAME)
    message(STATUS "HOST_SYSTEM_NAME=${HOST_SYSTEM_NAME}")

    if (${HOST_SYSTEM_NAME} STREQUAL "windows")
      # We only support x86_64
      set(HOST_SYSTEM_PROCESSOR "x86_64")
    else()
      execute_process(COMMAND uname -m OUTPUT_VARIABLE HOST_SYSTEM_PROCESSOR OUTPUT_STRIP_TRAILING_WHITESPACE)
      string(REPLACE arm aarch HOST_SYSTEM_PROCESSOR ${HOST_SYSTEM_PROCESSOR})
    endif()
    message(STATUS "HOST_SYSTEM_PROCESSOR=${HOST_SYSTEM_PROCESSOR}")
    set(CMAKE_TOOLCHAIN_FILE "cmake/${HOST_SYSTEM_NAME}-${HOST_SYSTEM_PROCESSOR}.cmake")
  endif()
endif()


project("ML")


include(variables)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  configure_file(${CMAKE_SOURCE_DIR}/cmake/Info.plist.in
    ${CMAKE_BINARY_DIR}/Info.plist
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
add_subdirectory(devbin)
add_subdirectory(devlib)

# Add a target to build Doxygen generated documentation
# if the doxygen executable can be found
ml_doxygen(${CMAKE_SOURCE_DIR}/build/doxygen)
