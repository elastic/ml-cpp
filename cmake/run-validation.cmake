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

# Portable CMake script that locates a Python 3 interpreter, ensures a
# virtual environment with the required packages exists, and then runs
# validate_allowlist.py.
#
# Required variables (passed via -D on command line):
#   SOURCE_DIR  - path to the repository root
#
# Optional variables:
#   VALIDATE_CONFIG  - path to validation_models.json
#   VALIDATE_PT_DIR  - directory of .pt files to validate
#   VALIDATE_VERBOSE - if TRUE, pass --verbose to the script
#   OPTIONAL         - if TRUE, skip gracefully when Python 3 is not
#                      found or dependency installation fails (instead
#                      of failing the build).  Intended for use when
#                      this script is invoked as part of a broader test
#                      target where the environment may not have Python
#                      or network access.

cmake_minimum_required(VERSION 3.19.2)

if(NOT DEFINED SOURCE_DIR)
  message(FATAL_ERROR "SOURCE_DIR must be defined")
endif()

# Helper: emit a FATAL_ERROR or a WARNING+return depending on OPTIONAL.
macro(_validation_fail _msg)
  if(DEFINED OPTIONAL AND OPTIONAL)
    message(WARNING "Skipping validation: ${_msg}")
    return()
  else()
    message(FATAL_ERROR "${_msg}")
  endif()
endmacro()

set(_tools_dir "${SOURCE_DIR}/dev-tools/extract_model_ops")
set(_venv_dir "${_tools_dir}/.venv")
set(_requirements "${_tools_dir}/requirements.txt")
set(_validate_script "${_tools_dir}/validate_allowlist.py")

# --- Locate a Python 3 interpreter ---
# Try names in order of preference.  On Linux build machines Python may
# only be available as python3.12 (installed via make altinstall).
# On Windows the canonical name is just "python".
find_program(_python_path
  NAMES python3 python3.12 python3.11 python3.10 python
  DOC "Python 3 interpreter (>= 3.10)"
)

if(NOT _python_path)
  _validation_fail(
    "No Python 3 interpreter found on PATH.\n"
    "Install Python 3 or ensure it is on your PATH.")
endif()

# Verify it is actually Python 3 (guards against "python" being Python 2).
execute_process(
  COMMAND "${_python_path}" --version
  OUTPUT_VARIABLE _py_version_out
  ERROR_VARIABLE _py_version_out
  RESULT_VARIABLE _py_rc
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _py_rc EQUAL 0 OR NOT _py_version_out MATCHES "Python 3\\.")
  _validation_fail(
    "Found ${_python_path} but it is not Python 3 (${_py_version_out}).")
endif()
message(STATUS "Found Python 3: ${_python_path} (${_py_version_out})")

# --- Platform-specific venv paths ---
if(CMAKE_HOST_WIN32)
  set(_venv_python "${_venv_dir}/Scripts/python.exe")
  set(_venv_pip "${_venv_dir}/Scripts/pip.exe")
else()
  set(_venv_python "${_venv_dir}/bin/python3")
  set(_venv_pip "${_venv_dir}/bin/pip")
endif()

# --- Create virtual environment if it does not exist ---
if(NOT EXISTS "${_venv_python}")
  message(STATUS "Creating virtual environment in ${_venv_dir} ...")
  execute_process(
    COMMAND "${_python_path}" -m venv "${_venv_dir}"
    RESULT_VARIABLE _venv_rc
  )
  if(NOT _venv_rc EQUAL 0)
    _validation_fail("Failed to create virtual environment (exit ${_venv_rc})")
  endif()
endif()

# --- Install / update dependencies when requirements.txt is newer ---
set(_stamp "${_venv_dir}/.requirements.stamp")
set(_needs_install FALSE)

if(NOT EXISTS "${_stamp}")
  set(_needs_install TRUE)
else()
  file(TIMESTAMP "${_requirements}" _req_ts "%Y%m%d%H%M%S" UTC)
  file(TIMESTAMP "${_stamp}" _stamp_ts "%Y%m%d%H%M%S" UTC)
  if(_req_ts STRGREATER _stamp_ts)
    set(_needs_install TRUE)
  endif()
endif()

if(_needs_install)
  message(STATUS "Installing/updating Python dependencies ...")
  execute_process(
    COMMAND "${_venv_pip}" install --quiet --upgrade pip
    RESULT_VARIABLE _pip_rc
  )
  if(NOT _pip_rc EQUAL 0)
    message(WARNING "pip upgrade failed (exit ${_pip_rc}) — continuing anyway")
  endif()

  execute_process(
    COMMAND "${_venv_pip}" install --quiet -r "${_requirements}"
    RESULT_VARIABLE _pip_rc
  )
  if(NOT _pip_rc EQUAL 0)
    _validation_fail(
      "Failed to install dependencies from ${_requirements} (exit ${_pip_rc}).\n"
      "This may indicate no network access is available.")
  endif()

  file(WRITE "${_stamp}" "installed")
endif()

# --- Ensure the venv's torch libraries take precedence ---
# When a locally-built libtorch is installed in a system path (e.g.
# /usr/local/lib on macOS), the pip-installed torch package's
# libtorch_python will pick up the wrong libtorch_cpu at load time.
# Prepending the venv's torch/lib directory to the dynamic library
# search path forces the pip-bundled libraries to be found first.
if(CMAKE_HOST_WIN32)
  set(_venv_site_packages "${_venv_dir}/Lib/site-packages")
else()
  # Query the venv Python for its site-packages directory rather than
  # globbing, which can yield a semicolon-separated list of paths.
  execute_process(
    COMMAND "${_venv_python}" -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE _venv_site_packages
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _sp_rc
  )
  if(NOT _sp_rc EQUAL 0 OR _venv_site_packages STREQUAL "")
    _validation_fail("Could not determine venv site-packages directory")
  endif()
endif()
set(_torch_lib_dir "${_venv_site_packages}/torch/lib")

if(EXISTS "${_torch_lib_dir}")
  if(CMAKE_HOST_APPLE)
    set(ENV{DYLD_LIBRARY_PATH} "${_torch_lib_dir}:$ENV{DYLD_LIBRARY_PATH}")
  elseif(NOT CMAKE_HOST_WIN32)
    set(ENV{LD_LIBRARY_PATH} "${_torch_lib_dir}:$ENV{LD_LIBRARY_PATH}")
  endif()
  message(STATUS "Prepended ${_torch_lib_dir} to dynamic library search path")
endif()

# --- Build the command line for validate_allowlist.py ---
set(_cmd "${_venv_python}" "${_validate_script}")

if(DEFINED VALIDATE_CONFIG)
  list(APPEND _cmd "--config" "${VALIDATE_CONFIG}")
endif()

if(DEFINED VALIDATE_PT_DIR)
  list(APPEND _cmd "--pt-dir" "${VALIDATE_PT_DIR}")
endif()

if(DEFINED VALIDATE_VERBOSE AND VALIDATE_VERBOSE)
  list(APPEND _cmd "--verbose")
endif()

message(STATUS "Running: ${_cmd}")

execute_process(
  COMMAND ${_cmd}
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE _validate_rc
)

if(NOT _validate_rc EQUAL 0)
  _validation_fail("Validation failed (exit ${_validate_rc})")
endif()
