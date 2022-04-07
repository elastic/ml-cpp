cmake_minimum_required(VERSION 3.7)

project("ML Test")

set (CMAKE_CXX_STANDARD 17)

find_package(Boost 1.77.0 REQUIRED COMPONENTS filesystem unit_test_framework)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

add_compile_definitions(BOOST_ALL_DYN_LINK BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
add_compile_definitions(RAPIDJSON_HAS_STDSTRING DRAPIDJSON_NEON)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
add_compile_definitions(MacOSX)
set(Boost_COMPILER -clang-darwin13)
set(CMAKE_MACOSX_RPATH 1)
set(PLATFORM_NAME "MacOSX")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
set(Boost_COMPILER gcc10)
set(PLATFORM_NAME "Linux")
endif()

list( APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib )


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

link_directories(${CMAKE_INSTALL_PREFIX}/lib)


add_library(MlTest SHARED ${SRCS} ${PLATFORM_SRCS})
target_link_libraries(MlTest ${Boost_LIBRARIES})
target_link_libraries(MlTest MlApi)
target_link_libraries(MlTest MlCore)
target_link_libraries(MlTest MlMathsAnalytics)
target_link_libraries(MlTest MlMathsCommon)

install(TARGETS MlTest DESTINATION lib)
