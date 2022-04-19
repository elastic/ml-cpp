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

project("ML Autodetect")

include_directories(../../../include)
include_directories(SYSTEM ../../../3rd_party/include)
include_directories(SYSTEM ../../../3rd_party/rapidjson/include)
include_directories(SYSTEM ../../../3rd_party/eigen)

set(SRCS
Main.cc
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(dump_state ${SRCS})

target_link_libraries(dump_state ${Boost_LIBRARIES})
target_link_libraries(dump_state libMlCore)
target_link_libraries(dump_state libMlApi)
target_link_libraries(dump_state libMlSeccomp)
target_link_libraries(dump_state libMlVer)

#install(TARGETS dump_state DESTINATION "${EXE_DIR}")
