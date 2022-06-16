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

set(ML_DEPENDENCIES eigen)
set(ML_LINK_LIBRARIES
  ${ZLIB_LIBRARIES}
  ${LIBXML2_LIBRARIES}
  ${Boost_LIBRARIES})

ml_add_library(MlCore SHARED
  CBase64Filter.cc
  CBlockingCallCancellerThread.cc
  CBlockingCallCancellingTimer.cc
  CCTimeR.cc
  CCompressOStream.cc
  CCompressedDictionary.cc
  CCondition.cc
  CContainerPrinter.cc
  CCrashHandler.cc
  CCsvLineParser.cc
  CDataAdder.cc
  CDataFrame.cc
  CDataFrameRowSlice.cc
  CDataSearcher.cc
  CDetachedProcessSpawner.cc
  CDualThreadStreamBuf.cc
  CFastMutex.cc
  CFlatPrefixTree.cc
  CGmTimeR.cc
  CHashing.cc
  CIEEE754.cc
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
  CMonotonicTime.cc
  CMutex.cc
  CNamedPipeFactory.cc
  COsFileFuncs.cc
  CPackedBitVector.cc
  CPatternSet.cc
  CPersistUtils.cc
  CProcess.cc
  CProcessPriority.cc
  CProgName.cc
  CProgramCounters.cc
  CRapidJsonConcurrentLineWriter.cc
  CRapidJsonUnbufferedIStreamWrapper.cc
  CRapidXmlParser.cc
  CRapidXmlStatePersistInserter.cc
  CRapidXmlStateRestoreTraverser.cc
  CReadWriteLock.cc
  CRegex.cc
  CRegexFilter.cc
  CResourceLocator.cc
  CScopedFastLock.cc
  CScopedLock.cc
  CScopedReadLock.cc
  CScopedWriteLock.cc
  CSetEnv.cc
  CSetMode.cc
  CShellArgQuoter.cc
  CStateCompressor.cc
  CStateDecompressor.cc
  CStateMachine.cc
  CStatePersistInserter.cc
  CStateRestoreTraverser.cc
  CStaticThreadPool.cc
  CStopWatch.cc
  CStoredStringPtr.cc
  CStrCaseCmp.cc
  CStrFTime.cc
  CStrPTime.cc
  CStrTokR.cc
  CStreamUtils.cc
  CStringCache.cc
  CStringSimilarityTester.cc
  CStringUtils.cc
  CThread.cc
  CTimeGm.cc
  CTimeUtils.cc
  CTimezone.cc
  CUnSetEnv.cc
  CUname.cc
  CWindowsError.cc
  CWordDictionary.cc
  CWordExtractor.cc
  CXmlNode.cc
  CXmlNodeWithChildren.cc
  CXmlNodeWithChildrenPool.cc
  CXmlParser.cc
  CXmlParserIntf.cc
  CompressUtils.cc
  Concurrency.cc
  )

if (WIN32)
  target_link_libraries(MlCore PUBLIC AdvAPI32.lib shell32.lib Version.lib ${STRPTIME_LIB})
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/date_time_zonespec.csv DESTINATION ${ML_RESOURCES_DIR})
endif()

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/ml-en.dict DESTINATION ${ML_RESOURCES_DIR})
