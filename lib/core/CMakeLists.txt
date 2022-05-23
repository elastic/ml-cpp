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

project("ML Core")

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/eigen)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)

add_compile_definitions(BUILDING_libMlCore)

set (PLATFORM_SRCS_TMP
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

gen_platform_srcs("${PLATFORM_SRCS_TMP}")

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
if (WIN32)
  target_link_libraries(MlCore PUBLIC AdvAPI32.lib shell32.lib Version.lib ${STRPTIME_LIB})
endif()

target_link_libraries(MlCore PUBLIC ${ZLIB_LIBRARIES})
target_link_libraries(MlCore PUBLIC ${LIBXML2_LIBRARIES})
target_link_libraries(MlCore PUBLIC ${Boost_LIBRARIES})

add_dependencies(MlCore eigen)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  target_link_libraries(MlCore PRIVATE "-current_version ${ML_VERSION_NUM}" "-compatibility_version ${ML_VERSION_NUM}" "${COVERAGE}")
endif()

install(TARGETS MlCore RUNTIME DESTINATION ${DYNAMIC_LIB_DIR} ARCHIVE DESTINATION ${IMPORT_LIB_DIR})

if(MSVC)
  install(FILES $<TARGET_PDB_FILE:MlCore> DESTINATION ${DYNAMIC_LIB_DIR} OPTIONAL)
  install(FILES $ENV{CPP_SRC_HOME}/lib/core/date_time_zonespec.csv DESTINATION ${CMAKE_INSTALL_PREFIX}/resources)
endif()
