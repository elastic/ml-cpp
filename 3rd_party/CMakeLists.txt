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

# Build Abseil and Sandbox2 on Linux only. MlSandbox (lib/sandbox) is a
# dormant target: it is built everywhere Sandbox2 is available, but nothing
# in the controller/pytorch_inference wiring routes to it yet. See
# docs/projects/mlcpp-sandbox2-pr2873 in the elastic-workspace harness for
# the staged-rollout contract this dependency foundation exists for.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  include(FetchContent)

  # Save and restore CMake cache state this block flips so it cannot change
  # the caller's build configuration for anything outside Sandbox2/Abseil.
  set(_saved_BUILD_TESTING ${BUILD_TESTING})
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(_saved_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

  # The vendored Abseil and Sandboxed API sources are not unity-build safe:
  # e.g. absl_time_zone defines kDigits in an anonymous namespace in both
  # time_zone_fixed.cc and time_zone_posix.cc, which collide when merged
  # into one unity translation unit. The top-level build configures
  # -DCMAKE_UNITY_BUILD=ON, so disable it for these third-party targets only.
  set(_saved_CMAKE_UNITY_BUILD ${CMAKE_UNITY_BUILD})
  set(CMAKE_UNITY_BUILD OFF)

  set(ABSL_PROPAGATE_CXX_STD ON CACHE INTERNAL "" FORCE)
  set(ABSL_USE_EXTERNAL_GOOGLETEST OFF CACHE INTERNAL "" FORCE)
  set(ABSL_FIND_GOOGLETEST OFF CACHE INTERNAL "" FORCE)
  set(ABSL_ENABLE_INSTALL OFF CACHE INTERNAL "" FORCE)
  set(ABSL_BUILD_TESTING OFF CACHE INTERNAL "" FORCE)
  set(ABSL_BUILD_TEST_HELPERS OFF CACHE INTERNAL "" FORCE)
  set(SAPI_ENABLE_EXAMPLES OFF CACHE INTERNAL "" FORCE)
  set(SAPI_ENABLE_TESTS OFF CACHE INTERNAL "" FORCE)

  set(ML_SANDBOXED_API_TAG v20241008)
  set(ML_SANDBOXED_API_PATCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/patches/sandboxed-api)
  set(ML_SANDBOXED_API_PATCHES
    0001-abseil-cpp-disable-gtest.patch
    0002-no-fno-exceptions-propagation.patch
    0003-python3-optional.patch
    0004-forkserver-zlib-static-libstdcxx.patch
  )

  FetchContent_Declare(
    sandboxed-api
    GIT_REPOSITORY https://github.com/google/sandboxed-api.git
    GIT_TAG ${ML_SANDBOXED_API_TAG}
    GIT_SHALLOW TRUE
  )

  FetchContent_GetProperties(sandboxed-api)
  if(NOT sandboxed-api_POPULATED)
    FetchContent_Populate(sandboxed-api)

    find_package(Git REQUIRED)
    foreach(_patch ${ML_SANDBOXED_API_PATCHES})
      # Re-running configure in an existing build directory can re-enter this
      # block even though the checked-out source was already patched in an
      # earlier configure (observed: FetchContent's populated-tracking does
      # not reliably short-circuit this across separate `cmake` invocations
      # on every CMake/generator combination). `git apply --check` alone
      # cannot distinguish "already applied" from "genuinely drifted" - both
      # fail to apply cleanly - so try a reverse-check first: if the patch
      # reverses cleanly, its change is already present and this is the
      # idempotent-rerun case, not drift.
      execute_process(
        COMMAND ${GIT_EXECUTABLE} apply --reverse --check ${ML_SANDBOXED_API_PATCH_DIR}/${_patch}
        WORKING_DIRECTORY ${sandboxed-api_SOURCE_DIR}
        RESULT_VARIABLE _patch_already_applied_result
        OUTPUT_QUIET
        ERROR_QUIET
      )
      if(_patch_already_applied_result EQUAL 0)
        message(STATUS "sandboxed-api patch already applied (reconfigure): ${_patch}")
        continue()
      endif()

      execute_process(
        COMMAND ${GIT_EXECUTABLE} apply --check ${ML_SANDBOXED_API_PATCH_DIR}/${_patch}
        WORKING_DIRECTORY ${sandboxed-api_SOURCE_DIR}
        RESULT_VARIABLE _patch_check_result
        OUTPUT_QUIET
        ERROR_VARIABLE _patch_check_error
      )
      if(NOT _patch_check_result EQUAL 0)
        message(FATAL_ERROR
          "sandboxed-api patch ${_patch} no longer applies to pinned tag "
          "${ML_SANDBOXED_API_TAG} - the upstream source has drifted since "
          "this patch was written. Regenerate it against the current tag "
          "content (see 3rd_party/patches/sandboxed-api/README.md).\n"
          "${_patch_check_error}")
      endif()
      execute_process(
        COMMAND ${GIT_EXECUTABLE} apply ${ML_SANDBOXED_API_PATCH_DIR}/${_patch}
        WORKING_DIRECTORY ${sandboxed-api_SOURCE_DIR}
        RESULT_VARIABLE _patch_apply_result
        ERROR_VARIABLE _patch_apply_error
      )
      if(NOT _patch_apply_result EQUAL 0)
        message(FATAL_ERROR "sandboxed-api patch ${_patch} failed to apply: ${_patch_apply_error}")
      endif()
      message(STATUS "Applied sandboxed-api patch: ${_patch}")
    endforeach()
  endif()

  add_subdirectory(${sandboxed-api_SOURCE_DIR} ${sandboxed-api_BINARY_DIR} EXCLUDE_FROM_ALL)

  if(TARGET sandbox2::sandbox2)
    set(SANDBOX2_LIBRARIES sandbox2::sandbox2 CACHE INTERNAL "Sandbox2 libraries")
    message(STATUS "Sandbox2 enabled: using sandbox2::sandbox2")
  else()
    message(FATAL_ERROR "Sandbox2 required on Linux but sandbox2::sandbox2 was not built")
  endif()

  # Restore the caller's settings for the rest of the build.
  set(BUILD_TESTING ${_saved_BUILD_TESTING} CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS ${_saved_BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)
  set(CMAKE_UNITY_BUILD ${_saved_CMAKE_UNITY_BUILD})
endif()
