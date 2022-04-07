cmake_minimum_required(VERSION 3.7)

project("ML Core lib")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake/Modules/")
set (CMAKE_CXX_STANDARD 17)

list( APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib )

#RAPIDJSONINCLUDES=-isystem $(CPP_SRC_HOME)/3rd_party/rapidjson/include
#ifeq ($(HARDWARE_ARCH),x86_64)
#RAPIDJSONCPPFLAGS=-DRAPIDJSON_HAS_STDSTRING -DRAPIDJSON_SSE42
#else
#RAPIDJSONCPPFLAGS=-DRAPIDJSON_HAS_STDSTRING -DRAPIDJSON_NEON
#endif

#if (MSVC)
#    # warning level 4 and all warnings as errors
#    add_compile_options(/W4 /WX)
#else()
#    # lots of warnings and all warnings as errors
#    add_compile_options(-Wall -Wextra -pedantic -Werror)
#endif()

#add_compile_definitions(-DRAPIDJSON_HAS_STDSTRING -DRAPIDJSON_NEON)
add_compile_definitions(RAPIDJSON_HAS_STDSTRING DRAPIDJSON_NEON)
add_compile_definitions(DEIGEN_MPL2_ONLY DEIGEN_MAX_ALIGN_BYTES=32)

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/eigen)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
set(Boost_COMPILER -clang-darwin13)
add_compile_definitions(MacOSX)
set(PLATFORM_SRCS
CCondition.cc
CCrashHandler.cc
CCTimeR.cc
CDetachedProcessSpawner.cc
CFastMutex_MacOSX.cc
CGmTimeR.cc
CIEEE754.cc
CMonotonicTime_MacOSX.cc
CMutex.cc
CNamedPipeFactory.cc
COsFileFuncs.cc
CProcess.cc
CProcessPriority.cc
CProgName_MacOSX.cc
CReadWriteLock.cc
CSetEnv.cc
CSetMode.cc
CShellArgQuoter.cc
CStateMachine.cc
CStrCaseCmp.cc
CStrFTime.cc
CStrPTime.cc
CStrTokR.cc
CThread.cc
CTimeGm.cc
CTimezone.cc
CUname.cc
CUnSetEnv.cc
CWindowsError.cc
)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
set(Boost_COMPILER gcc10)
set(PLATFORM_SRCS
CCondition.cc
CCrashHandler_Linux.cc
CCTimeR.cc
CDetachedProcessSpawner.cc
CFastMutex.cc
CGmTimeR.cc
CIEEE754.cc
CMonotonicTime.cc
CMutex.cc
CNamedPipeFactory.cc
COsFileFuncs.cc
CProcess.cc
CProcessPriority_Linux.cc
CProgName_Linux.cc
CReadWriteLock.cc
CSetEnv.cc
CSetMode.cc
CShellArgQuoter.cc
CStateMachine.cc
CStrCaseCmp.cc
CStrFTime.cc
CStrPTime_Linux.cc
CStrTokR.cc
CThread.cc
CTimeGm.cc
CTimezone.cc
CUname.cc
CUnSetEnv.cc
CWindowsError.cc
)
endif()

set(SRCS
CBase64Filter.cc 
CBlockingCallCancellerThread.cc
CBlockingCallCancellingTimer.cc
CCompressedDictionary.cc
CCompressOStream.cc
CContainerPrinter.cc
CCsvLineParser.cc
CDataAdder.cc
CDataFrame.cc
CDataFrameRowSlice.cc
CDataSearcher.cc
CDualThreadStreamBuf.cc
CFlatPrefixTree.cc
CHashing.cc
CJsonLogLayout.cc
CJsonOutputStreamWrapper.cc
CJsonStatePersistInserter.cc
CJsonStateRestoreTraverser.cc
CLogger.cc
CLoggerThrottler.cc
CLoopProgress.cc
CMemory.cc
CMemoryUsage.cc
CMemoryUsageJsonWriter.cc
CompressUtils.cc
Concurrency.cc
CPackedBitVector.cc
CPatternSet.cc
CPersistUtils.cc
CProgramCounters.cc
CRapidJsonConcurrentLineWriter.cc
CRapidJsonUnbufferedIStreamWrapper.cc
CRapidXmlParser.cc
CRapidXmlStatePersistInserter.cc
CRapidXmlStateRestoreTraverser.cc
CRegex.cc
CRegexFilter.cc
CResourceLocator.cc
CScopedFastLock.cc
CScopedLock.cc
CScopedReadLock.cc
CScopedWriteLock.cc
CStateCompressor.cc
CStateDecompressor.cc
CStatePersistInserter.cc
CStateRestoreTraverser.cc
CStaticThreadPool.cc
CStopWatch.cc
CStreamUtils.cc
CStoredStringPtr.cc
CStringCache.cc
CStringSimilarityTester.cc
CStringUtils.cc
CTimeUtils.cc
CWordDictionary.cc
CWordExtractor.cc
CXmlNode.cc
CXmlNodeWithChildren.cc
CXmlNodeWithChildrenPool.cc
CXmlParser.cc
CXmlParserIntf.cc
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


add_library(MlCore SHARED ${SRCS} ${PLATFORM_SRCS})
target_link_libraries(MlCore ${Boost_LIBRARIES})
target_link_libraries(MlCore ${ZLIB_LIBRARIES})
target_link_libraries(MlCore ${LIBXML2_LIBRARIES})

install(TARGETS MlCore DESTINATION lib)
