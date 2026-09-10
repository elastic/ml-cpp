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

ml_add_executable(controller
  CBlockingCallCancellingStreamMonitor.cc
  CCmdLineParser.cc
  CCommandProcessor.cc
  CProcessSpawnerRouter.cc
  CResponseJsonWriter.cc
  )

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
