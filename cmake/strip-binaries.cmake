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

# strip-binaries.cmake
#
# Portable replacement for dev-tools/strip_binaries.sh.
# Strips ML native code binaries to reduce download size.
# Works on macOS and Linux (including cross-compilation); no-op on Windows.
#
# Required: CPP_PLATFORM_HOME (env var or -D parameter)
# Optional: CPP_CROSS_COMPILE (env var) for Linux cross-compilation

cmake_minimum_required(VERSION 3.19)

if(CMAKE_HOST_WIN32)
  message(STATUS "Stripping not required on Windows (symbols are in .pdb files)")
  return()
endif()

# ---------------------------------------------------------------------------
# Validate CPP_PLATFORM_HOME
# ---------------------------------------------------------------------------
if(NOT DEFINED CPP_PLATFORM_HOME)
  if(DEFINED ENV{CPP_PLATFORM_HOME})
    set(CPP_PLATFORM_HOME "$ENV{CPP_PLATFORM_HOME}")
  else()
    message(FATAL_ERROR "CPP_PLATFORM_HOME must be defined")
  endif()
endif()

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
if(CMAKE_HOST_APPLE)
  set(_exe_dir "${CPP_PLATFORM_HOME}/controller.app/Contents/MacOS")
  set(_lib_dir "${CPP_PLATFORM_HOME}/controller.app/Contents/lib")
else()
  set(_exe_dir "${CPP_PLATFORM_HOME}/bin")
  set(_lib_dir "${CPP_PLATFORM_HOME}/lib")
endif()

if(NOT IS_DIRECTORY "${_exe_dir}")
  message(FATAL_ERROR "${_exe_dir} does not exist")
endif()
if(NOT IS_DIRECTORY "${_lib_dir}")
  message(FATAL_ERROR "${_lib_dir} does not exist")
endif()

# ---------------------------------------------------------------------------
# Find required tools
# ---------------------------------------------------------------------------
if(CMAKE_HOST_APPLE)
  find_program(_dsymutil dsymutil REQUIRED)
  find_program(_strip strip REQUIRED)
elseif(DEFINED ENV{CPP_CROSS_COMPILE})
  set(_cross_prefix "$ENV{CPP_CROSS_COMPILE}-linux-gnu")
  find_program(_objcopy ${_cross_prefix}-objcopy REQUIRED)
  find_program(_strip ${_cross_prefix}-strip REQUIRED)
else()
  find_program(_objcopy objcopy REQUIRED)
  find_program(_strip strip REQUIRED)
endif()

# ---------------------------------------------------------------------------
# macOS stripping helpers
# ---------------------------------------------------------------------------
function(_strip_macos_executable _prog)
  message(STATUS "Stripping ${_prog}")
  execute_process(COMMAND "${_dsymutil}" "${_prog}" RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(WARNING "dsymutil failed for ${_prog}")
  endif()
  execute_process(COMMAND "${_strip}" -u -r "${_prog}" RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(WARNING "strip failed for ${_prog}")
  endif()
endfunction()

function(_strip_macos_library _lib)
  message(STATUS "Stripping ${_lib}")
  get_filename_component(_name "${_lib}" NAME)
  if(_name MATCHES "Ml")
    execute_process(COMMAND "${_dsymutil}" "${_lib}" RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(WARNING "dsymutil failed for ${_lib}")
    endif()
  endif()
  execute_process(COMMAND "${_strip}" -x "${_lib}" RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(WARNING "strip failed for ${_lib}")
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Linux stripping helpers
# ---------------------------------------------------------------------------
function(_strip_linux_executable _prog)
  message(STATUS "Stripping ${_prog}")
  execute_process(COMMAND "${_objcopy}" --only-keep-debug "${_prog}" "${_prog}-debug"
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(WARNING "objcopy --only-keep-debug failed for ${_prog}")
    return()
  endif()
  execute_process(COMMAND "${_strip}" --strip-all "${_prog}" RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(WARNING "strip failed for ${_prog}")
  endif()
  execute_process(COMMAND "${_objcopy}" "--add-gnu-debuglink=${_prog}-debug" "${_prog}"
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(WARNING "objcopy --add-gnu-debuglink failed for ${_prog}")
  endif()
  file(CHMOD "${_prog}-debug" PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
endfunction()

function(_strip_linux_library _lib)
  message(STATUS "Stripping ${_lib}")
  execute_process(COMMAND "${_objcopy}" --only-keep-debug "${_lib}" "${_lib}-debug"
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(WARNING "objcopy --only-keep-debug failed for ${_lib}")
    return()
  endif()
  execute_process(COMMAND "${_strip}" --strip-unneeded "${_lib}" RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(WARNING "strip failed for ${_lib}")
  endif()
  execute_process(COMMAND "${_objcopy}" "--add-gnu-debuglink=${_lib}-debug" "${_lib}"
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(WARNING "objcopy --add-gnu-debuglink failed for ${_lib}")
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Strip executables
# ---------------------------------------------------------------------------
file(GLOB _executables "${_exe_dir}/*")
foreach(_exe IN LISTS _executables)
  if(IS_DIRECTORY "${_exe}")
    continue()
  endif()
  get_filename_component(_name "${_exe}" NAME)
  if(_name MATCHES "\\.dSYM$" OR _name MATCHES "-debug$" OR _name STREQUAL "core")
    continue()
  endif()
  if(CMAKE_HOST_APPLE)
    _strip_macos_executable("${_exe}")
  else()
    _strip_linux_executable("${_exe}")
  endif()
endforeach()

# ---------------------------------------------------------------------------
# Strip libraries
# ---------------------------------------------------------------------------
file(GLOB _libraries "${_lib_dir}/*")
foreach(_lib IN LISTS _libraries)
  if(IS_DIRECTORY "${_lib}")
    continue()
  endif()
  get_filename_component(_name "${_lib}" NAME)
  if(_name MATCHES "\\.dSYM$" OR _name MATCHES "-debug$")
    continue()
  endif()
  if(CMAKE_HOST_APPLE)
    _strip_macos_library("${_lib}")
  else()
    _strip_linux_library("${_lib}")
  endif()
endforeach()
