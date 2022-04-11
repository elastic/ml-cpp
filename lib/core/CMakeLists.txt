project("ML Core lib")

list( APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib )

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/eigen)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)

set(PLATFORM_SRCS_TMP
CCondition.cc
CCrashHandler.cc
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
CProcessPriority.cc
CProgName.cc
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

add_library(MlCore SHARED ${SRCS} ${PLATFORM_SRCS})
target_link_libraries(MlCore PUBLIC ${Boost_LIBRARIES})
target_link_libraries(MlCore PUBLIC ${ZLIB_LIBRARIES})
target_link_libraries(MlCore PUBLIC ${LIBXML2_LIBRARIES})

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(MlCore PRIVATE "-current_version ${ML_VERSION_NUM}" "-compatibility_version ${ML_VERSION_NUM}" "${COVERAGE}")
endif()

install(TARGETS MlCore DESTINATION lib)
