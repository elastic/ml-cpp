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

project("ML Sandbox unit tests")

set(SRCS
  Main.cc
  CMlSandboxAvailabilityTest.cc
  )

set(ML_LINK_LIBRARIES
  ${Boost_LIBRARIES_WITH_UNIT_TEST}
  MlCore
  MlSandbox
  MlTest
  )

if(TARGET sandbox2::sandbox2 AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # The forkserver runtime smoke test links the Sandbox2 API directly (not
  # just MlSandbox, which exposes no sandbox2 symbols yet) to prove the
  # vendored forkserver - built via 3rd_party/patches/sandboxed-api/ - can
  # fork/exec/reap a real child. It is Linux-only and dropped entirely
  # elsewhere rather than compiled out with #ifdef, since sandbox2 headers
  # are unavailable on non-Linux configure runs.
  list(APPEND SRCS CSandboxForkserverSmokeTest.cc)
  list(APPEND ML_LINK_LIBRARIES sandbox2::sandbox2)

  # Deliberately-dependency-free sandboxee payload for the smoke test above.
  # Built with plain add_executable rather than ml_add_non_distributed_executable:
  # it has no ml-cpp library dependencies, no ML_LINK_LIBRARIES, and must not
  # be confused with a distributable ml-cpp binary. Dynamically linked (the
  # default) - a static build was tried first to sidestep
  # PolicyBuilder::AddLibrariesForBinary(), matching upstream sandboxed-api's
  # own examples/static/static_bin.cc, but the ml-cpp CI build image
  # (docker.elastic.co/ml-dev/ml-linux-build) has no static libc/libm
  # archives (`ld: cannot find -lm/-lc`), so the smoke test itself now calls
  # AddLibrariesForBinary() on the dynamically-linked payload instead.
  add_executable(sandbox2_smoke_payload EXCLUDE_FROM_ALL
    payloads/sandbox_smoke_payload.cc
    )
  set_target_properties(sandbox2_smoke_payload PROPERTIES
    POSITION_INDEPENDENT_CODE TRUE
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/payloads
    )
endif()

ml_add_test_executable(sandbox ${SRCS})

if(TARGET sandbox2_smoke_payload)
  add_dependencies(ml_test_sandbox sandbox2_smoke_payload)
  target_compile_definitions(ml_test_sandbox PRIVATE
    ML_SANDBOX2_SMOKE_PAYLOAD="$<TARGET_FILE:sandbox2_smoke_payload>"
    )
endif()
