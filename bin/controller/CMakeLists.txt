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

add_library(MlController OBJECT ${SRCS})

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(controller Main.cc $<TARGET_OBJECTS:MlController>)

target_link_libraries(controller ${Boost_LIBRARIES})
target_link_libraries(controller MlCore)
target_link_libraries(controller MlVer)
target_link_libraries(controller MlSeccomp)

install(TARGETS controller DESTINATION "${EXE_DIR}")
if(MSVC)
  set( controller_LINKFLAGS ${CMAKE_CURRENT_BINARY_DIR}/controller.res )
  set_target_properties( controller PROPERTIES LINK_FLAGS ${controller_LINKFLAGS} )

  execute_process(COMMAND bash -c "${CMAKE_SOURCE_DIR}/mk/make_rc_defines.sh controller.exe" OUTPUT_VARIABLE RC_DEFINES)
  string(STRIP ${RC_DEFINES} RC_DEFINES)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh "rc -nologo ${CPPFLAGS} ${RC_DEFINES} -Focontroller.res ${CMAKE_SOURCE_DIR}/mk/ml.rc")
  add_custom_target(
      controller.res
      DEPENDS ${CMAKE_SOURCE_DIR}/mk/ml.rc ${CMAKE_SOURCE_DIR}/gradle.properties ${CMAKE_SOURCE_DIR}/mk/ml.ico ${CMAKE_SOURCE_DIR}/mk/make_rc_defines.sh ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh 
      COMMAND bash -c ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh
  )

  add_dependencies(controller controller.res)
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh)
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/controller.res)

  install(FILES $<TARGET_PDB_FILE:controller> DESTINATION ${EXE_DIR} OPTIONAL)
endif()
