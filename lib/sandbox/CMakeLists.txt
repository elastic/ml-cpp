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

project("ML Sandbox")

set(ML_LINK_LIBRARIES
  MlCore
  MlSeccomp
  )

set(SRCS
  CSandbox2Diagnostics.cc
  CSandboxedProcessSpawner.cc
  CPytorchInferenceSandboxPolicy.cc
  )

ml_add_library(MlSandbox STATIC ${SRCS})

if(TARGET sandbox2::sandbox2)
  target_compile_definitions(MlSandbox PUBLIC SANDBOX2_AVAILABLE)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # libunwind_ptrace (sandboxed-api) references uncompress. MlCore may already
    # place -lz earlier on consumer link lines; --as-needed can drop it before the
    # archives on aarch64. PUBLIC + --no-as-needed keeps -lz after sandbox2 for
    # controller and ml_test_sandbox. sandbox2::sandbox2 is an ALIAS target, so
    # zlib cannot be attached via target_link_libraries on that alias in 3rd_party.
    target_link_libraries(MlSandbox PUBLIC sandbox2::sandbox2)
    target_link_options(MlSandbox PUBLIC "LINKER:--push-state,--no-as-needed")
    target_link_libraries(MlSandbox PUBLIC ${ZLIB_LIBRARIES})
    target_link_options(MlSandbox PUBLIC "LINKER:--pop-state")
    message(STATUS "MlSandbox: Sandbox2 enabled and linked")
  endif()
endif()
