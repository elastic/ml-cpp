cmake_minimum_required(VERSION 3.7)

project("ML Core lib unit tests")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake/Modules/")
set (CMAKE_CXX_STANDARD 17)

include_directories(../../../include)
include_directories(SYSTEM ../../../3rd_party/include)
include_directories(SYSTEM ../../../3rd_party/eigen)
include_directories(SYSTEM ../../../3rd_party/rapidjson/include)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
set(Boost_COMPILER -clang-darwin13)
add_compile_definitions(MacOSX)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
set(Boost_COMPILER gcc10)
endif()

add_compile_definitions(RAPIDJSON_HAS_STDSTRING)
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Darwin detected")
  # darwin flags
  set(CMAKE_C_FLAGS "-g ${OPT_C_FLAGS} ${ARCHFLAGS} -fstack-protector -Weverything -Werror-switch -Wno-deprecated -Wno-disabled-macro-expansion -Wno-documentation-deprecated-sync -Wno-documentation-unknown-command -Wno-float-equal -Wno-missing-prototypes -Wno-padded -Wno-poison-system-directories -Wno-sign-conversion -Wno-unknown-warning-option -Wno-unreachable-code -Wno-used-but-marked-unused" )
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-exit-time-destructors -Wno-global-constructors -Wno-return-std-move-in-c++11 -Wno-unused-member-function -Wno-weak-vtables" )
  set(Boost_COMPILER -clang-darwin13)
  set(EXE_DIR MacOS)
  set(CMAKE_MACOSX_RPATH 1)
  set(MacOSX)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Linux detected")
  # linux flags
  set(CMAKE_C_FLAGS "-g ${OPTCFLAGS} ${ARCHCFLAGS} -fstack-protector -fno-math-errno -Wall -Wcast-align -Wconversion -Wextra -Winit-self -Wno-psabi -Wparentheses -Wpointer-arith -Wswitch-enum")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-ctor-dtor-privacy -Wno-deprecated-declarations -Wold-style-cast -fvisibility-inlines-hidden")
  
  execute_process(COMMAND uname -m OUTPUT_VARIABLE HARDWARE_ARCH)
  
  if(HARDWARE_ARCH STREQUAL "aarch64")
    message(STATUS "aarch64 detected")
    set(ARCHCFLAGS "-march=armv8-a+crc+crypto")
    add_compile_definitions(DRAPIDJSON_NEON)
  else()
    message(STATUS "x86-64 detected")
    set (ARCHCFLAGS "-msse4.2 -mfpmath=sse")
    add_compile_definitions(DRAPIDJSON_SSE42)
  endif()
  
  set(Boost_COMPILER gcc10)
  set(EXE_DIR bin)
endif()


# TODO make platform safe
set (PLATFORM_SRCS
CProcessPriorityTest.cc
)

set (SRCS
Main.cc 
CAlignmentTest.cc 
CAllocationStrategyTest.cc 
CBase64FilterTest.cc 
CBlockingCallCancellingTimerTest.cc 
CCompressedDictionaryTest.cc 
CCompressUtilsTest.cc 
CConcurrencyTest.cc 
CConcurrentWrapperTest.cc 
CContainerPrinterTest.cc 
CContainerThroughputTest.cc 
CCsvLineParserTest.cc 
CDataFrameTest.cc 
CDetachedProcessSpawnerTest.cc 
CDualThreadStreamBufTest.cc 
CFlatPrefixTreeTest.cc 
CFunctionalTest.cc 
CHashingTest.cc 
CIEEE754Test.cc 
CJsonLogLayoutTest.cc 
CJsonOutputStreamWrapperTest.cc 
CJsonStatePersistInserterTest.cc 
CJsonStateRestoreTraverserTest.cc 
CLoggerTest.cc 
CLoggerThrottlerTest.cc 
CLoopProgressTest.cc 
CMemoryUsageJsonWriterTest.cc 
CMemoryUsageTest.cc 
CMonotonicTimeTest.cc 
CMutexTest.cc 
CNamedPipeFactoryTest.cc 
COsFileFuncsTest.cc 
CPackedBitVectorTest.cc 
CPatternSetTest.cc 
CPersistenceTagTest.cc 
CPersistUtilsTest.cc 
CProcessTest.cc 
CProgNameTest.cc 
CProgramCountersTest.cc 
CRapidJsonLineWriterTest.cc
CRapidJsonWriterBaseTest.cc
CRapidJsonUnbufferedIStreamWrapperTest.cc 
CRapidXmlParserTest.cc 
CRapidXmlStatePersistInserterTest.cc 
CRapidXmlStateRestoreTraverserTest.cc 
CReadWriteLockTest.cc 
CRegexFilterTest.cc 
CRegexTest.cc 
CResourceLocatorTest.cc 
CShellArgQuoterTest.cc 
CSmallVectorTest.cc 
CStateCompressorTest.cc 
CStateMachineTest.cc 
CStaticThreadPoolTest.cc 
CStopWatchTest.cc 
CStoredStringPtrTest.cc 
CStringSimilarityTesterTest.cc 
CStringUtilsTest.cc 
CThreadMutexConditionTest.cc 
CTimeUtilsTest.cc 
CTripleTest.cc 
CUnameTest.cc 
CVectorRangeTest.cc 
CWindowsErrorTest.cc 
CWordDictionaryTest.cc 
CWordExtractorTest.cc 
CXmlNodeWithChildrenTest.cc 
CXmlParserTest.cc 
)

find_package(Boost 1.77.0 REQUIRED COMPONENTS iostreams filesystem program_options regex date_time log_setup thread unit_test_framework)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

find_package(LibXml2)
if(LIBXML2_FOUND)
  include_directories(${LIBXML2_INCLUDE_DIR})
endif()

find_package(ZLIB)
if(ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIR})
endif()

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(ml_test_core ${SRCS} ${PLATFORM_SRCS})

target_link_libraries(ml_test_core ${Boost_LIBRARIES})
target_link_libraries(ml_test_core ${LIBXML2_LIBRARIES})
target_link_libraries(ml_test_core MlMathsCommon)
target_link_libraries(ml_test_core MlCore)
target_link_libraries(ml_test_core MlTest)

add_test(ml_test_core ml_test_core)
# make "make test" run the test program
add_custom_target(test_core
     DEPENDS ml_test_core
     COMMAND ml_test_core
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
