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

project("ML Normalize")

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/eigen)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)

set(SRCS
Main.cc
CCmdLineParser.cc
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(normalize ${SRCS})

target_link_libraries(normalize ${Boost_LIBRARIES})
target_link_libraries(normalize MlCore)
target_link_libraries(normalize MlModel)
target_link_libraries(normalize MlApi)
target_link_libraries(normalize MlVer)
target_link_libraries(normalize MlSeccomp)

install(TARGETS normalize DESTINATION "${EXE_DIR}")
if(MSVC)
  set( normalize_LINKFLAGS ${CMAKE_CURRENT_BINARY_DIR}/normalize.res )
  set_target_properties( normalize PROPERTIES LINK_FLAGS ${normalize_LINKFLAGS} )
  set_property(TARGET normalize PROPERTY COMPILE_PDB_NAME normalize)

  execute_process(COMMAND bash -c "${CMAKE_SOURCE_DIR}/mk/make_rc_defines.sh normalize.exe" OUTPUT_VARIABLE RC_DEFINES)
  string(STRIP ${RC_DEFINES} RC_DEFINES)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh "rc -nologo ${CPPFLAGS} ${RC_DEFINES} -Fonormalize.res ${CMAKE_SOURCE_DIR}/mk/ml.rc")
  add_custom_target(
      normalize.res
      DEPENDS ${CMAKE_SOURCE_DIR}/mk/ml.rc ${CMAKE_SOURCE_DIR}/gradle.properties ${CMAKE_SOURCE_DIR}/mk/ml.ico ${CMAKE_SOURCE_DIR}/mk/make_rc_defines.sh ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh 
      COMMAND bash -c ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh
  )

  add_dependencies(normalize normalize.res)
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh)
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/normalize.res)

  install(FILES $<TARGET_PDB_FILE:normalize> DESTINATION ${EXE_DIR} OPTIONAL)
endif()
