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
  set( data_frame_analyzer_LINKFLAGS ${CMAKE_CURRENT_BINARY_DIR}/data_frame_analyzer.res )
  set_target_properties( data_frame_analyzer PROPERTIES LINK_FLAGS ${data_frame_analyzer_LINKFLAGS} )

  execute_process(COMMAND bash -c "${CMAKE_SOURCE_DIR}/mk/make_rc_defines.sh data_frame_analyzer.exe" OUTPUT_VARIABLE RC_DEFINES)
  string(STRIP ${RC_DEFINES} RC_DEFINES)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh "rc -nologo ${CPPFLAGS} ${RC_DEFINES} -Fodata_frame_analyzer.res ${CMAKE_SOURCE_DIR}/mk/ml.rc")
  add_custom_target(
      data_frame_analyzer.res
      DEPENDS ${CMAKE_SOURCE_DIR}/mk/ml.rc ${CMAKE_SOURCE_DIR}/gradle.properties ${CMAKE_SOURCE_DIR}/mk/ml.ico ${CMAKE_SOURCE_DIR}/mk/make_rc_defines.sh ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh 
      COMMAND bash -c ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh
  )

  add_dependencies(data_frame_analyzer data_frame_analyzer.res)
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh)
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/data_frame_analyzer.res)
 
  install(FILES $<TARGET_PDB_FILE:data_frame_analyzer> DESTINATION ${EXE_DIR} OPTIONAL)
endif()
