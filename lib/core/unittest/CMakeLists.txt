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

project("ML Core unit tests")

set (SRCS
    CAlignmentTest.cc 
    CAllocationStrategyTest.cc 
    CBase64FilterTest.cc 
    CBlockingCallCancellingTimerTest.cc 
    CCompressUtilsTest.cc 
    CCompressedDictionaryTest.cc 
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
    CPersistUtilsTest.cc 
    CPersistenceTagTest.cc 
    CProcessPriorityTest.cc
    CProcessTest.cc 
    CProgNameTest.cc 
    CProgramCountersTest.cc 
    CRapidJsonLineWriterTest.cc
    CRapidJsonUnbufferedIStreamWrapperTest.cc 
    CRapidJsonWriterBaseTest.cc
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
    Main.cc 
)

set(ML_LINK_LIBRARIES
    ${Boost_LIBRARIES_WITH_UNIT_TEST}
    ${LIBXML2_LIBRARIES}
    MlCore
    MlMathsCommon
    MlTest
)

ml_add_test_executable(core ${SRCS})
