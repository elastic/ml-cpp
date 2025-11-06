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

# Install 3rd party libraries in our build tree
# as part of the CMake configuration step - avoiding
# the need for it to be done on every build
execute_process(
    COMMAND ${CMAKE_COMMAND} -DINSTALL_DIR=${INSTALL_DIR} -DCMAKE_CXX_COMPILER_VERSION_MAJOR=${CMAKE_CXX_COMPILER_VERSION_MAJOR} -P ./3rd_party.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Pull the Eigen repo as part of the configuration step
# thus avoiding any race conditions with parallel builds
execute_process(
    COMMAND ${CMAKE_COMMAND} -P ./pull-eigen.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Pull the Valijson repo as part of the configuration step
# thus avoiding any race conditions with parallel builds
execute_process(
    COMMAND ${CMAKE_COMMAND} -P ./pull-valijson.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Build Abseil and Sandbox2 on Linux only
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Include FetchContent module
  include(FetchContent)
  
  # Save and disable BUILD_TESTING to prevent CTest from enabling tests
  set(_saved_BUILD_TESTING ${BUILD_TESTING})
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  
  # Disable Google Test-related options to avoid dependency issues
  # Set as regular variables first to ensure they're available during FetchContent
  set(ABSL_PROPAGATE_CXX_STD ON)
  set(ABSL_USE_EXTERNAL_GOOGLETEST OFF)
  set(ABSL_FIND_GOOGLETEST OFF)
  set(ABSL_ENABLE_INSTALL OFF)
  set(ABSL_BUILD_TESTING OFF)
  set(ABSL_BUILD_TEST_HELPERS OFF)
  
  # Sandboxed API options
  set(SAPI_ENABLE_EXAMPLES OFF)
  set(SAPI_ENABLE_TESTS OFF)
  
  # Also cache them to ensure they persist
  set(ABSL_PROPAGATE_CXX_STD ON CACHE INTERNAL "" FORCE)
  set(ABSL_USE_EXTERNAL_GOOGLETEST OFF CACHE INTERNAL "" FORCE)
  set(ABSL_FIND_GOOGLETEST OFF CACHE INTERNAL "" FORCE)
  set(ABSL_ENABLE_INSTALL OFF CACHE INTERNAL "" FORCE)
  set(ABSL_BUILD_TESTING OFF CACHE INTERNAL "" FORCE)
  set(ABSL_BUILD_TEST_HELPERS OFF CACHE INTERNAL "" FORCE)
  set(SAPI_ENABLE_EXAMPLES OFF CACHE INTERNAL "" FORCE)
  set(SAPI_ENABLE_TESTS OFF CACHE INTERNAL "" FORCE)
  
  # Declare Sandboxed API dependency
  FetchContent_Declare(
    sandboxed-api
    GIT_REPOSITORY https://github.com/google/sandboxed-api.git
    GIT_TAG v20241008
    GIT_SHALLOW TRUE
  )
  
  # Get the source directory before making it available
  FetchContent_GetProperties(sandboxed-api)
  if(NOT sandboxed-api_POPULATED)
    FetchContent_Populate(sandboxed-api)
    
    # Override the abseil-cpp.cmake file to disable Google Test
    file(WRITE ${sandboxed-api_SOURCE_DIR}/cmake/abseil-cpp.cmake
"# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FetchContent_Declare(absl
  URL https://github.com/abseil/abseil-cpp/archive/61e47a454c81eb07147b0315485f476513cc1230.zip  # 2024-04-05
  URL_HASH SHA256=9ba0e97acf7026f7479e24967866ba9560cf3256304b6c8932d2b1ab7d0dfcd2
)
set(ABSL_CXX_STANDARD \${SAPI_CXX_STANDARD} CACHE STRING \"\" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL \"\" FORCE)
set(ABSL_RUN_TESTS OFF CACHE BOOL \"\" FORCE)
set(ABSL_BUILD_TEST_HELPERS OFF CACHE BOOL \"\" FORCE)
set(ABSL_USE_EXTERNAL_GOOGLETEST OFF)
set(ABSL_FIND_GOOGLETEST OFF)
set(ABSL_USE_GOOGLETEST_HEAD OFF CACHE BOOL \"\" FORCE)

FetchContent_MakeAvailable(absl)
")
  endif()
  
  # Patch sandboxed-api CMakeLists.txt to remove -fno-exceptions PUBLIC flag
  # This prevents it from propagating to our codebase
  file(READ ${sandboxed-api_SOURCE_DIR}/CMakeLists.txt _sapi_cmake_content)
  string(REPLACE 
    "target_compile_options(sapi_base PUBLIC\n  -fno-exceptions\n)"
    "# target_compile_options(sapi_base PUBLIC\n#   -fno-exceptions\n# )"
    _sapi_cmake_content "${_sapi_cmake_content}")
  file(WRITE ${sandboxed-api_SOURCE_DIR}/CMakeLists.txt "${_sapi_cmake_content}")
  
  # Patch SapiDeps.cmake to make Python3 optional
  # Python3 is only needed for protobuf code generation, which may not be required
  # if protobuf is already built or if we're using pre-generated code
  if(EXISTS ${sandboxed-api_SOURCE_DIR}/cmake/SapiDeps.cmake)
    file(READ ${sandboxed-api_SOURCE_DIR}/cmake/SapiDeps.cmake _sapi_deps_content)
    
    # Step 1: Make find_package(Python3) non-REQUIRED by adding QUIET
    # This prevents CMake from erroring if Python3 is not found
    string(REPLACE
      "find_package(Python3 REQUIRED"
      "find_package(Python3 QUIET"
      _sapi_deps_content "${_sapi_deps_content}")
    string(REPLACE
      "find_package(Python3 COMPONENTS Interpreter"
      "find_package(Python3 QUIET COMPONENTS Interpreter"
      _sapi_deps_content "${_sapi_deps_content}")
    string(REPLACE
      "find_package(Python3\n"
      "find_package(Python3 QUIET\n"
      _sapi_deps_content "${_sapi_deps_content}")
    
    # Step 2: Make find_package_handle_standard_args safe
    # The issue is that find_package_handle_standard_args with REQUIRED_VARS will fail
    # if Python3_EXECUTABLE is not set. Instead of trying to match complex multiline
    # patterns (which is fragile), we use a simpler approach:
    # 1. Ensure Python3_EXECUTABLE is always defined (even if empty) before any checks
    # 2. Remove REQUIRED flag from find_package_handle_standard_args calls
    # 3. Wrap the call in a conditional that only runs if Python3_EXECUTABLE is set
    #
    # This approach is more robust because it doesn't depend on exact pattern matching
    # for closing parentheses or multiline structures.
    
    # Ensure Python3_EXECUTABLE is defined before any find_package_handle_standard_args calls
    # Insert this right after the find_package(Python3) call
    string(REGEX REPLACE
      "(find_package\\(Python3[^\n]*\\))"
      "\\1\n# Ensure Python3_EXECUTABLE is defined to prevent REQUIRED_VARS from failing\nif(NOT DEFINED Python3_EXECUTABLE)\n  set(Python3_EXECUTABLE \"\")\nendif()"
      _sapi_deps_content "${_sapi_deps_content}")
    
    # Remove REQUIRED flag from find_package_handle_standard_args calls
    string(REGEX REPLACE
      "find_package_handle_standard_args\\(Python3[^)]*\\bREQUIRED\\b[^)]*\\)"
      "find_package_handle_standard_args(Python3)"
      _sapi_deps_content "${_sapi_deps_content}")
    
    # Wrap find_package_handle_standard_args calls that use REQUIRED_VARS in a conditional
    # This prevents the fatal error when Python3_EXECUTABLE is empty
    # Handle multiline patterns by matching across newlines using (.|\n) pattern
    # First, handle the common multiline pattern with Interpreter
    string(REGEX REPLACE
      "([ \t]*)find_package_handle_standard_args\\(Python3([^)]*REQUIRED_VARS[^)]*Python3_EXECUTABLE[^)]*Interpreter[^)]*)\\)"
      "\\1# Python3 check (made optional)\n\\1if(Python3_EXECUTABLE)\n\\1  find_package_handle_standard_args(Python3\\2)\n\\1else()\n\\1  set(Python3_FOUND FALSE)\n\\1  message(STATUS \"Python3 not found - continuing without it (protobuf code generation may be disabled)\")\n\\1endif()"
      _sapi_deps_content "${_sapi_deps_content}")
    
    # Handle case with just Python3_EXECUTABLE (no Interpreter)
    string(REGEX REPLACE
      "([ \t]*)find_package_handle_standard_args\\(Python3([^)]*REQUIRED_VARS[^)]*Python3_EXECUTABLE[^)]*)\\)"
      "\\1# Python3 check (made optional)\n\\1if(Python3_EXECUTABLE)\n\\1  find_package_handle_standard_args(Python3\\2)\n\\1else()\n\\1  set(Python3_FOUND FALSE)\n\\1  message(STATUS \"Python3 not found - continuing without it (protobuf code generation may be disabled)\")\n\\1endif()"
      _sapi_deps_content "${_sapi_deps_content}")
    
    file(WRITE ${sandboxed-api_SOURCE_DIR}/cmake/SapiDeps.cmake "${_sapi_deps_content}")
  endif()
  
  # Now add the subdirectory
  add_subdirectory(${sandboxed-api_SOURCE_DIR} ${sandboxed-api_BINARY_DIR} EXCLUDE_FROM_ALL)
  
  # Set SANDBOX2_LIBRARIES after sandboxed-api is available
  # Check if sandbox2::sandbox2 target exists (it should if sandboxed-api built successfully)
  if(TARGET sandbox2::sandbox2)
    set(SANDBOX2_LIBRARIES sandbox2::sandbox2 CACHE INTERNAL "Sandbox2 libraries")
    message(STATUS "Sandbox2 enabled: using sandbox2::sandbox2")
  else()
    set(SANDBOX2_LIBRARIES "" CACHE INTERNAL "Sandbox2 libraries not available")
    message(WARNING "Sandbox2 target not found - Sandbox2 will be disabled")
  endif()
  
  # Restore BUILD_TESTING if it was set
  if(_saved_BUILD_TESTING)
    set(BUILD_TESTING ${_saved_BUILD_TESTING} CACHE BOOL "" FORCE)
  endif()
endif()
