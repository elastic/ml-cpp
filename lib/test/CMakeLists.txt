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

project("ML Test")

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/eigen)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)

add_compile_definitions(BUILDING_libMlTest)

set(PLATFORM_SRCS_TMP
CShellCmdEscape.cc
)

foreach(FILE ${PLATFORM_SRCS_TMP})
    set(FILE "${CMAKE_CURRENT_SOURCE_DIR}/${FILE}")
    string(REPLACE ".cc" "_${PLATFORM_NAME}.cc" FILE_TMP ${FILE})
    if (EXISTS ${FILE_TMP})
        list(APPEND PLATFORM_SRCS ${FILE_TMP})
    else()
        list(APPEND PLATFORM_SRCS ${FILE})
    endif()
endforeach()

set(SRCS
CBoostTestXmlOutput.cc
CDataFrameAnalysisSpecificationFactory.cc
CDataFrameAnalyzerTrainingFactory.cc
CMultiFileDataAdder.cc
CMultiFileSearcher.cc
CProgramCounterClearingFixture.cc
CRandomNumbers.cc
CTestObserver.cc
CTestTmpDir.cc
CThreadDataReader.cc
CThreadDataWriter.cc
CTimeSeriesTestData.cc
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)


add_library(libMlTest SHARED ${SRCS} ${PLATFORM_SRCS})
target_link_libraries(libMlTest PUBLIC ${Boost_LIBRARIES})
target_link_libraries(libMlTest PUBLIC libMlApi)
target_link_libraries(libMlTest PUBLIC libMlCore)
target_link_libraries(libMlTest PUBLIC libMlMathsAnalytics)
target_link_libraries(libMlTest PUBLIC libMlMathsCommon)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(libMlTest PRIVATE "-current_version ${ML_VERSION_NUM}" "-compatibility_version ${ML_VERSION_NUM}" "${COVERAGE}")
endif()

install(TARGETS libMlTest RUNTIME DESTINATION ${DYNAMIC_LIB_DIR} ARCHIVE DESTINATION ${IMPORT_LIB_DIR})
install(FILES $<TARGET_PDB_FILE:libMlTest> DESTINATION ${DYNAMIC_LIB_DIR} OPTIONAL)
