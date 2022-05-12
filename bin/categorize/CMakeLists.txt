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
  set( categorize_LINKFLAGS ${CMAKE_CURRENT_BINARY_DIR}/categorize.res )
  set_target_properties( categorize PROPERTIES LINK_FLAGS ${categorize_LINKFLAGS} )

  execute_process(COMMAND bash -c "${CMAKE_SOURCE_DIR}/mk/make_rc_defines.sh categorize.exe" OUTPUT_VARIABLE RC_DEFINES)
  string(STRIP ${RC_DEFINES} RC_DEFINES)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh "rc -nologo ${RC_FLAGS} ${RC_DEFINES} -Focategorize.res ${CMAKE_SOURCE_DIR}/mk/ml.rc")
  add_custom_target(
      categorize.res
      DEPENDS ${CMAKE_SOURCE_DIR}/mk/ml.rc ${CMAKE_SOURCE_DIR}/gradle.properties ${CMAKE_SOURCE_DIR}/mk/ml.ico ${CMAKE_SOURCE_DIR}/mk/make_rc_defines.sh ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh 
      COMMAND bash -c ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh
  )

  add_dependencies(categorize categorize.res)
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh)
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/categorize.res)

  install(FILES $<TARGET_PDB_FILE:categorize> DESTINATION ${EXE_DIR} OPTIONAL)
endif()
