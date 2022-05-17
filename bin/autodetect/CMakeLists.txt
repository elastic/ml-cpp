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

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)
include_directories(SYSTEM ../../3rd_party/eigen)

set(SRCS
Main.cc
CCmdLineParser.cc
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(autodetect ${SRCS})

# TODO - Not a final solution by any means
list(REMOVE_ITEM Boost_LIBRARIES "Boost::unit_test_framework")

target_link_libraries(autodetect ${Boost_LIBRARIES})
target_link_libraries(autodetect MlCore)
target_link_libraries(autodetect MlMathsCommon)
target_link_libraries(autodetect MlMathsTimeSeries)
target_link_libraries(autodetect MlModel)
target_link_libraries(autodetect MlApi)
target_link_libraries(autodetect MlSeccomp)
target_link_libraries(autodetect MlVer)

install(TARGETS autodetect DESTINATION "${EXE_DIR}")
if(MSVC)
  set( autodetect_LINKFLAGS ${CMAKE_CURRENT_BINARY_DIR}/autodetect.res )
  set_target_properties( autodetect PROPERTIES LINK_FLAGS ${autodetect_LINKFLAGS} )

  execute_process(COMMAND bash -c "${CMAKE_SOURCE_DIR}/mk/make_rc_defines.sh autodetect.exe" OUTPUT_VARIABLE RC_DEFINES)
  string(STRIP ${RC_DEFINES} RC_DEFINES)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh "rc -nologo ${CPPFLAGS} ${RC_DEFINES} -Foautodetect.res ${CMAKE_SOURCE_DIR}/mk/ml.rc")
  add_custom_target(
      autodetect.res
      DEPENDS ${CMAKE_SOURCE_DIR}/mk/ml.rc ${CMAKE_SOURCE_DIR}/gradle.properties ${CMAKE_SOURCE_DIR}/mk/ml.ico ${CMAKE_SOURCE_DIR}/mk/make_rc_defines.sh ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh
      COMMAND bash -c ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh
  )
  add_dependencies(autodetect autodetect.res)

  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh)
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/autodetect.res)

  install(FILES $<TARGET_PDB_FILE:autodetect> DESTINATION ${EXE_DIR} OPTIONAL)
endif()
