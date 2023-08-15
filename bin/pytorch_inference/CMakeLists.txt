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

project("ML PyTorch Inference")

# IPEX is required on Linux x86_64 architecture
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(LINK_IPEX ON)
endif()

if (LINK_IPEX)
  find_package(intel_ext_pt_cpu REQUIRED)

  set(ML_LINK_LIBRARIES 
    ${Boost_LIBRARIES}
    MlCore
    MlApi
    MlSeccomp
    MlVer
    ${TORCH_LIB}
    ${C10_LIB}
    ${IPEX_LIB} 
    )
else()
  set(ML_LINK_LIBRARIES 
    ${Boost_LIBRARIES}
    MlCore
    MlApi
    MlSeccomp
    MlVer
    ${TORCH_LIB}
    ${C10_LIB}
    )
endif()

ml_add_executable(pytorch_inference
  CBufferedIStreamAdapter.cc
  CCmdLineParser.cc
  CCommandParser.cc
  CResultWriter.cc
  CThreadSettings.cc
  )

