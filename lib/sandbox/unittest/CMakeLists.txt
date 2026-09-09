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
  # be confused with a distributable ml-cpp binary. Statically linked (as
  # upstream sandboxed-api's own examples/static/static_bin.cc is) so the
  # sandboxee has no dynamic-loader dependency for Sandbox2 to mount -
  # PolicyBuilder::AddLibrariesForBinary() ldd-mounting is real production
  # policy work that belongs to a later PR, not this dormant-foundation smoke
  # test.
  add_executable(sandbox2_smoke_payload EXCLUDE_FROM_ALL
    payloads/sandbox_smoke_payload.cc
    )
  set_target_properties(sandbox2_smoke_payload PROPERTIES
    POSITION_INDEPENDENT_CODE FALSE
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/payloads
    LINK_FLAGS "-static"
    )
endif()

ml_add_test_executable(sandbox ${SRCS})

if(TARGET sandbox2_smoke_payload)
  add_dependencies(ml_test_sandbox sandbox2_smoke_payload)
  target_compile_definitions(ml_test_sandbox PRIVATE
    ML_SANDBOX2_SMOKE_PAYLOAD="$<TARGET_FILE:sandbox2_smoke_payload>"
    )
endif()
