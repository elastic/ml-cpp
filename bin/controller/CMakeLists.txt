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

project("ML Controller")

set(ML_LINK_LIBRARIES
  ${Boost_LIBRARIES}
  MlCore
  MlSandbox
  MlSeccomp
  MlVer
  )

# CProcessSpawnerRouter.cc is the only controller source that includes
# Sandbox2/Abseil/protobuf headers (via CSandboxedProcessSpawner.h). Those
# headers are not warning-clean under ml-cpp's strict flags; under the debug
# CI build's CMAKE_COMPILE_WARNING_AS_ERROR=ON they fail compilation when
# built as part of the controller target. Compile it in a dedicated static
# library with warnings-as-errors off, matching lib/sandbox/CMakeLists.txt for
# MlSandbox.
add_library(MlProcessSpawnerRouter STATIC CProcessSpawnerRouter.cc)
set_target_properties(MlProcessSpawnerRouter PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  COMPILE_WARNING_AS_ERROR OFF)
target_link_libraries(MlProcessSpawnerRouter PUBLIC MlCore MlSandbox)

ml_add_executable(controller
  CBlockingCallCancellingStreamMonitor.cc
  CCmdLineParser.cc
  CCommandProcessor.cc
  CResponseJsonWriter.cc
  )

target_link_libraries(controller PRIVATE MlProcessSpawnerRouter)

# ml_add_executable() also creates an OBJECT library (Mlcontroller) holding
# the sources above, purely so bin/controller/unittest can link the same
# object files as the executable. That OBJECT library has no link libraries
# of its own, so - unlike the `controller` executable target - it does not
# inherit MlSandbox's usage requirements, and in particular does not see
# MlSandbox's PUBLIC SANDBOX2_AVAILABLE compile definition. The unit test
# executable *does* link MlSandbox and therefore does see it, so without
# this line ml_test_controller mixes two different views of
# include/sandbox/CSandboxedProcessSpawner.h in one binary: that header
# declares one extra member (the m_AwaitResultFn seam) under
# SANDBOX2_AVAILABLE, so sizeof(CSandboxedProcessSpawner) - and hence
# sizeof(CProcessSpawnerRouter) and sizeof(CCommandProcessor) - differ
# between the object files and the test translation units. That is an ODR
# violation, and it corrupted memory during test teardown on Linux.
# Link the OBJECT library against MlSandbox so its sources are compiled
# with exactly the same Sandbox2 configuration as both the production
# executable and the unit tests.
if(TARGET Mlcontroller)
  target_link_libraries(Mlcontroller PRIVATE MlSandbox)
endif()
