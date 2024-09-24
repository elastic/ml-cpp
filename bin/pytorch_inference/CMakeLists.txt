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

set(ML_LINK_LIBRARIES 
  ${Boost_LIBRARIES}
  MlCore
  MlApi
  MlSeccomp
  MlVer
  ${TORCH_LIB}
  ${C10_LIB}
  )

if (LINK_TCMALLOC)
  message(AUTHOR_WARNING "Linking libtcmalloc. Build is not for production release.")
  list(APPEND ML_LINK_LIBRARIES tcmalloc)
endif ()

if (LINK_PROFILER)
  message(AUTHOR_WARNING "Linking libprofiler. Build is not for production release.")
  list(APPEND ML_LINK_LIBRARIES profiler)
endif ()

set(MKL_ARCH intel64)
set(MKL_THREADING sequential)
set(MKL_INTERFACE lp64)
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
message(STATUS "Imported oneMKL targets: ${MKL_IMPORTED_TARGETS}")


ml_add_executable(pytorch_inference
  CBufferedIStreamAdapter.cc
  CCmdLineParser.cc
  CCommandParser.cc
  CResultWriter.cc
  CThreadSettings.cc
  )

target_compile_options(pytorch_inference PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(pytorch_inference PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(pytorch_inference PUBLIC $<LINK_ONLY:MKL::MKL>)

ml_codesign(pytorch_inference)
