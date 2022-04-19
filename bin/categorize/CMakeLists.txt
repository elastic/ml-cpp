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

project("ML Categorize")

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)
include_directories(SYSTEM ../../3rd_party/eigen)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

set(SRCS
Main.cc
CCmdLineParser.cc
)

add_executable(categorize ${SRCS})

target_link_libraries(categorize ${Boost_LIBRARIES})
target_link_libraries(categorize MlCore)
target_link_libraries(categorize MlMathsCommon)
target_link_libraries(categorize MlModel)
target_link_libraries(categorize MlApi)
target_link_libraries(categorize MlSeccomp)
target_link_libraries(categorize MlVer)

install(TARGETS categorize DESTINATION "${EXE_DIR}")
if(MSVC)
  install(FILES $<TARGET_PDB_FILE:categorize> DESTINATION ${EXE_DIR} OPTIONAL)
endif()
