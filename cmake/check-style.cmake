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

# check-style.cmake
#
# Portable replacement for dev-tools/check-style.sh.
# Checks clang-format compliance and copyright headers for C++ source files.
#
# Optional -D parameters:
#   CHECK_ALL  - ON to check all files, OFF to check only git-changed files (default: OFF)
#   SOURCE_DIR - root of the source tree (default: CPP_SRC_HOME env var, or git toplevel)

cmake_minimum_required(VERSION 3.19)

# ---------------------------------------------------------------------------
# Determine source directory
# ---------------------------------------------------------------------------
if(NOT DEFINED SOURCE_DIR OR "${SOURCE_DIR}" STREQUAL "")
  if(DEFINED ENV{CPP_SRC_HOME})
    set(SOURCE_DIR "$ENV{CPP_SRC_HOME}")
  else()
    execute_process(
      COMMAND git rev-parse --show-toplevel
      OUTPUT_VARIABLE SOURCE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0 OR "${SOURCE_DIR}" STREQUAL "")
      message(FATAL_ERROR "Could not determine source directory. Set SOURCE_DIR or CPP_SRC_HOME.")
    endif()
  endif()
endif()

if(NOT DEFINED CHECK_ALL)
  set(CHECK_ALL OFF)
endif()

# ---------------------------------------------------------------------------
# Find and validate clang-format
# ---------------------------------------------------------------------------
find_program(CLANG_FORMAT clang-format)
if(NOT CLANG_FORMAT)
  message(FATAL_ERROR "The clang-format code formatter is not available.")
endif()

set(_required_version "5.0.1")
execute_process(
  COMMAND "${CLANG_FORMAT}" --version
  OUTPUT_VARIABLE _version_output
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(_version_output MATCHES "([0-9]+\\.[0-9]+\\.[0-9]+)")
  set(_found_version "${CMAKE_MATCH_1}")
else()
  message(FATAL_ERROR "Could not determine clang-format version.")
endif()

if(NOT "${_found_version}" STREQUAL "${_required_version}")
  message(FATAL_ERROR "Required clang-format version ${_required_version}, found ${_found_version}")
endif()

# ---------------------------------------------------------------------------
# Read copyright header template
# ---------------------------------------------------------------------------
set(_copyright_file "${SOURCE_DIR}/copyright_code_header.txt")
if(NOT EXISTS "${_copyright_file}")
  message(FATAL_ERROR "Copyright header file not found: ${_copyright_file}")
endif()
file(READ "${_copyright_file}" _copyright_header)
string(LENGTH "${_copyright_header}" _copyright_len)

# ---------------------------------------------------------------------------
# Discover source files
# ---------------------------------------------------------------------------
set(_source_files "")

if(CHECK_ALL)
  file(GLOB_RECURSE _all_files
    "${SOURCE_DIR}/lib/*.cc"  "${SOURCE_DIR}/lib/*.h"
    "${SOURCE_DIR}/include/*.h"
    "${SOURCE_DIR}/bin/*.cc"  "${SOURCE_DIR}/bin/*.h"
    "${SOURCE_DIR}/devinclude/*.h")
  foreach(_f IN LISTS _all_files)
    # Exclude 3rd_party and build-setup directories
    if(_f MATCHES "/3rd_party/" OR _f MATCHES "/build-setup/")
      continue()
    endif()
    list(APPEND _source_files "${_f}")
  endforeach()
else()
  execute_process(
    COMMAND git diff --name-only --diff-filter=ACMRT
    WORKING_DIRECTORY "${SOURCE_DIR}"
    OUTPUT_VARIABLE _git_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rc)
  if(_rc EQUAL 0 AND NOT "${_git_output}" STREQUAL "")
    string(REPLACE "\n" ";" _git_files "${_git_output}")
    foreach(_f IN LISTS _git_files)
      if(_f MATCHES "/3rd_party/")
        continue()
      endif()
      if(_f MATCHES "\\.(cc|h)$")
        set(_full "${SOURCE_DIR}/${_f}")
        if(EXISTS "${_full}")
          message(STATUS "Checking: ${_f}")
          list(APPEND _source_files "${_full}")
        endif()
      endif()
    endforeach()
  endif()
endif()

list(LENGTH _source_files _num_files)
if(_num_files EQUAL 0)
  message(STATUS "No source files to check.")
  return()
endif()
message(STATUS "Checking ${_num_files} file(s)...")

# ---------------------------------------------------------------------------
# Check each file
# ---------------------------------------------------------------------------
set(_format_errors "")
set(_copyright_errors "")

foreach(_file IN LISTS _source_files)
  # Check clang-format compliance
  execute_process(
    COMMAND "${CLANG_FORMAT}" "${_file}"
    OUTPUT_VARIABLE _formatted
    RESULT_VARIABLE _rc)
  if(_rc EQUAL 0)
    file(READ "${_file}" _original)
    if(NOT "${_original}" STREQUAL "${_formatted}")
      file(RELATIVE_PATH _rel "${SOURCE_DIR}" "${_file}")
      list(APPEND _format_errors "${_rel}")
    endif()
  endif()

  # Check copyright header
  file(READ "${_file}" _file_content)
  string(SUBSTRING "${_file_content}" 0 ${_copyright_len} _file_header)
  if(NOT "${_file_header}" STREQUAL "${_copyright_header}")
    file(RELATIVE_PATH _rel "${SOURCE_DIR}" "${_file}")
    list(APPEND _copyright_errors "${_rel}")
  endif()
endforeach()

# ---------------------------------------------------------------------------
# Report results
# ---------------------------------------------------------------------------
set(_rc 0)

list(LENGTH _format_errors _num_format_errors)
if(_num_format_errors GREATER 0)
  message("")
  message("A format error has been detected within the following files:")
  foreach(_f IN LISTS _format_errors)
    message("  ${_f}")
  endforeach()
  set(_rc 1)
else()
  message(STATUS "No format errors detected")
endif()

list(LENGTH _copyright_errors _num_copyright_errors)
if(_num_copyright_errors GREATER 0)
  message("")
  message("The following files do not contain the correct copyright header:")
  foreach(_f IN LISTS _copyright_errors)
    message("  ${_f}")
  endforeach()
  set(_rc 1)
else()
  message(STATUS "No copyright header errors detected")
endif()

if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "Style check failed")
endif()
