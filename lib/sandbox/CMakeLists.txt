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

# MlSandbox links Sandbox2/Abseil and builds a runnable Sandbox2 forkserver
# on Linux, and now a typed filesystem/network launch policy for a
# pytorch_inference child. No controller or pytorch_inference routing
# depends on it yet - the process spawner and controller wiring land in
# follow-up PRs.

project("ML Sandbox")

set(ML_LINK_LIBRARIES
  MlCore
  MlSeccomp
  )

set(SRCS
  CMlSandboxAvailability.cc
  CPytorchInferenceSandboxPolicy.cc
  CSandboxedProcessSpawner_Linux.cc
  )

ml_add_library(MlSandbox STATIC ${SRCS})

if(TARGET sandbox2::sandbox2)
  target_compile_definitions(MlSandbox PUBLIC SANDBOX2_AVAILABLE)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # libunwind_ptrace (a Sandbox2 dependency, via sandbox2::unwind) calls
    # uncompress() from libz. CMake may de-duplicate ZLIB::ZLIB with MlCore
    # and place -lz before the sandbox2 archives on the link line; with
    # --as-needed that silently drops -lz from downstream executable links
    # (observed on aarch64). A raw -Wl group is a distinct link item, so -lz
    # stays ordered after sandbox2 regardless of de-duplication.
    # sandbox2::sandbox2 is an ALIAS target - do not target_link_libraries
    # against the alias name from outside this cache variable.
    target_link_libraries(MlSandbox PUBLIC sandbox2::sandbox2)
    target_link_libraries(MlSandbox PUBLIC "-Wl,--no-as-needed,-lz,--as-needed")
    message(STATUS "MlSandbox: Sandbox2 enabled and linked")
  endif()
else()
  message(STATUS "MlSandbox: Sandbox2 not available on this platform - building dormant stub only")
endif()
