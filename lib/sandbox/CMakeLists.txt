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
    target_link_libraries(MlSandbox PRIVATE sandbox2::sandbox2 ${ZLIB_LIBRARIES})
    message(STATUS "MlSandbox: Sandbox2 enabled and linked")
  endif()
endif()
