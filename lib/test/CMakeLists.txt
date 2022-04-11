project("ML Test")

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/eigen)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)

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


add_library(MlTest SHARED ${SRCS} ${PLATFORM_SRCS})
target_link_libraries(MlTest PUBLIC ${Boost_LIBRARIES})
target_link_libraries(MlTest PUBLIC MlApi)
target_link_libraries(MlTest PUBLIC MlCore)
target_link_libraries(MlTest PUBLIC MlMathsAnalytics)
target_link_libraries(MlTest PUBLIC MlMathsCommon)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(MlTest PRIVATE "-current_version ${ML_VERSION_NUM}" "-compatibility_version ${ML_VERSION_NUM}" "${COVERAGE}")
endif()

install(TARGETS MlTest DESTINATION lib)
