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

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)
include_directories(SYSTEM ../../3rd_party/eigen)

set(SRCS
CBlockingCallCancellingStreamMonitor.cc
CCmdLineParser.cc
CCommandProcessor.cc
CResponseJsonWriter.cc
)

add_library(MlController STATIC ${SRCS})
target_link_libraries(MlController ${Boost_LIBRARIES})

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(controller Main.cc)

target_link_libraries(controller MlController)
target_link_libraries(controller ${Boost_LIBRARIES})
target_link_libraries(controller MlCore)
target_link_libraries(controller MlVer)
target_link_libraries(controller MlSeccomp)

install(TARGETS controller DESTINATION "${EXE_DIR}")
if(MSVC)
  install(FILES $<TARGET_PDB_FILE:controller> DESTINATION ${EXE_DIR} OPTIONAL)
endif()
