cmake_minimum_required(VERSION 3.7)

project("ML Test")

add_compile_definitions(RAPIDJSON_HAS_STDSTRING DRAPIDJSON_NEON MacOSX)

include_directories(../../include)
include_directories(../../3rd_party/include)
include_directories(../../3rd_party/eigen)
include_directories(../../3rd_party/rapidjson/include)

set(PLATFORM_SRCS
CShellCmdEscape.cc
)

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

add_library(MlTest SHARED ${SRCS} ${PLATFORM_SRCS})
target_link_libraries(MlTest ${Boost_LIBRARIES})
target_link_libraries(MlTest MlApi)
target_link_libraries(MlTest MlCore)
target_link_libraries(MlTest MlMathsAnalytics)
target_link_libraries(MlTest MlMathsCommon)

install(TARGETS MlTest DESTINATION lib)
