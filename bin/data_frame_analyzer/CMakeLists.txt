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

project("ML DataFrame Analyzer")

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/eigen)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)

set(SRCS
Main.cc
CCmdLineParser.cc
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(data_frame_analyzer ${SRCS})

target_link_libraries(data_frame_analyzer ${Boost_LIBRARIES})
target_link_libraries(data_frame_analyzer MlCore)
target_link_libraries(data_frame_analyzer MlMathsCommon)
target_link_libraries(data_frame_analyzer MlMathsAnalytics)
target_link_libraries(data_frame_analyzer MlModel)
target_link_libraries(data_frame_analyzer MlApi)
target_link_libraries(data_frame_analyzer MlSeccomp)
target_link_libraries(data_frame_analyzer MlVer)

install(TARGETS data_frame_analyzer DESTINATION "${EXE_DIR}")
if(MSVC)
  install(FILES $<TARGET_PDB_FILE:data_frame_analyzer> DESTINATION ${EXE_DIR} OPTIONAL)
endif()
