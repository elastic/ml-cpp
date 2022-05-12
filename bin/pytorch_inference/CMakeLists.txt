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

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

set(SRCS
CBufferedIStreamAdapter.cc
CCmdLineParser.cc
CCommandParser.cc
CThreadSettings.cc
)

add_library(MlPytorchInference OBJECT ${SRCS})

add_executable(pytorch_inference Main.cc $<TARGET_OBJECTS:MlPytorchInference>)

target_link_libraries(pytorch_inference ${Boost_LIBRARIES})
target_link_libraries(pytorch_inference MlCore)
target_link_libraries(pytorch_inference MlApi)
target_link_libraries(pytorch_inference MlSeccomp)
target_link_libraries(pytorch_inference MlVer)
target_link_libraries(pytorch_inference "${TORCH_LIB}")
target_link_libraries(pytorch_inference "${C10_LIB}")

install(TARGETS pytorch_inference DESTINATION "${EXE_DIR}")
if(MSVC)
  set( pytorch_inference_LINKFLAGS ${CMAKE_CURRENT_BINARY_DIR}/pytorch_inference.res )
  set_target_properties( pytorch_inference PROPERTIES LINK_FLAGS ${pytorch_inference_LINKFLAGS} )

  execute_process(COMMAND bash -c "${CMAKE_SOURCE_DIR}/mk/make_rc_defines.sh pytorch_inference.exe" OUTPUT_VARIABLE RC_DEFINES)
  string(STRIP ${RC_DEFINES} RC_DEFINES)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh "rc -nologo ${RC_FLAGS} ${RC_DEFINES} -Fopytorch_inference.res ${CMAKE_SOURCE_DIR}/mk/ml.rc")
  add_custom_target(
      pytorch_inference.res
      DEPENDS ${CMAKE_SOURCE_DIR}/mk/ml.rc ${CMAKE_SOURCE_DIR}/gradle.properties ${CMAKE_SOURCE_DIR}/mk/ml.ico ${CMAKE_SOURCE_DIR}/mk/make_rc_defines.sh ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh 
      COMMAND bash -c ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh
  )

  add_dependencies(pytorch_inference pytorch_inference.res)
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh)
 
  install(FILES $<TARGET_PDB_FILE:pytorch_inference> DESTINATION ${EXE_DIR} OPTIONAL)
endif()
