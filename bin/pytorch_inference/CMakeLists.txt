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
Main.cc
CBufferedIStreamAdapter.cc
CCmdLineParser.cc
CCommandParser.cc
SettingsValidator.cc
)

add_executable(pytorch_inference ${SRCS})

target_link_libraries(pytorch_inference ${Boost_LIBRARIES})
target_link_libraries(pytorch_inference libMlCore)
target_link_libraries(pytorch_inference libMlApi)
target_link_libraries(pytorch_inference libMlSeccomp)
target_link_libraries(pytorch_inference libMlVer)
target_link_libraries(pytorch_inference "${TORCH_LIB}")
target_link_libraries(pytorch_inference "${C10_LIB}")
#set_property(TARGET  pytorch_inference PROPERTY CXX_STANDARD 14)

install(TARGETS pytorch_inference DESTINATION "${EXE_DIR}")
install(FILES $<TARGET_PDB_FILE:pytorch_inference> DESTINATION ${EXE_DIR} OPTIONAL)
