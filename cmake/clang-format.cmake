#!/bin/bash
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

# Reformats Ml native source code, using clang-format,  to ensure consistency.

# Set ${CMAKE_SOURCE_DIR} to the CPP_SRC_HOME environment variable
# if that is available, else set it to the topleve of the git repo.
if(DEFINED ENV{CPP_SRC_HOME})
  set(CMAKE_SOURCE_DIR $ENV{CPP_SRC_HOME})
else()
  execute_process(COMMAND git rev-parse --show-toplevel OUTPUT_VARIABLE CMAKE_SOURCE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

# Ensure clang-format is available and is the correct version
find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT STREQUAL "CLANG_FORMAT-NOTFOUND")
  message(FATAL_ERROR "The clang-format code formatter is not available. Exiting.")
  return()
endif()

set(REQUIRED_CLANG_FORMAT_VERSION 5.0.1)
execute_process(COMMAND ${CLANG_FORMAT} --version OUTPUT_VARIABLE FOUND_CLANG_FORMAT_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
if(${FOUND_CLANG_FORMAT_VERSION} MATCHES "version \([0-9].[0-9].[0-9]\)")
  set(FOUND_CLANG_FORMAT_VERSION ${CMAKE_MATCH_1})
endif()

if (NOT FOUND_CLANG_FORMAT_VERSION)
  message(STATUS "Could not determine clang-format version.")
  message(FATAL_ERROR "ERROR: Required clang-format major version ${REQUIRED_CLANG_FORMAT_VERSION} not found.")
  return()
endif()

if(NOT ${REQUIRED_CLANG_FORMAT_VERSION} VERSION_EQUAL ${FOUND_CLANG_FORMAT_VERSION})
  message(STATUS "Detected clang-format version ${FOUND_CLANG_FORMAT_VERSION}")
  message(FATAL_ERROR "Required clang-format major version ${REQUIRED_CLANG_FORMAT_VERSION} not found.")
  return()
endif()

file(GLOB_RECURSE SOURCE_FILES 
  ${CMAKE_SOURCE_DIR}/lib/*.cc
  ${CMAKE_SOURCE_DIR}/include/*.h
  ${CMAKE_SOURCE_DIR}/bin/*.cc
  ${CMAKE_SOURCE_DIR}/bin/*.h)
foreach(SOURCE_FILE ${SOURCE_FILES})
  execute_process(COMMAND ${CLANG_FORMAT} -i ${SOURCE_FILE})
endforeach()
