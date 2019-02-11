/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <test/CTestRunner.h>

#include "CAllocationStrategyTest.h"
#include "CBase64FilterTest.h"
#include "CBlockingMessageQueueTest.h"
#include "CByteSwapperTest.h"
#include "CCompressUtilsTest.h"
#include "CCompressedDictionaryTest.h"
#include "CConcurrencyTest.h"
#include "CConcurrentWrapperTest.h"
#include "CContainerPrinterTest.h"
#include "CContainerThroughputTest.h"
#include "CDataFrameTest.h"
#include "CDelimiterTest.h"
#include "CDetachedProcessSpawnerTest.h"
#include "CDualThreadStreamBufTest.h"
#include "CFileDeleterTest.h"
#include "CFlatPrefixTreeTest.h"
#include "CFunctionalTest.h"
#include "CHashingTest.h"
#include "CHexUtilsTest.h"
#include "CIEEE754Test.h"
#include "CJsonLogLayoutTest.h"
#include "CJsonOutputStreamWrapperTest.h"
#include "CJsonStatePersistInserterTest.h"
#include "CJsonStateRestoreTraverserTest.h"
#include "CLoggerTest.h"
#include "CLoopProgressTest.h"
#include "CMapPopulationTest.h"
#include "CMemoryUsageJsonWriterTest.h"
#include "CMemoryUsageTest.h"
#include "CMessageBufferTest.h"
#include "CMessageQueueTest.h"
#include "CMonotonicTimeTest.h"
#include "CMutexTest.h"
#include "CNamedPipeFactoryTest.h"
#include "COsFileFuncsTest.h"
#include "CPatternSetTest.h"
#include "CPersistUtilsTest.h"
#include "CPolymorphicStackObjectCPtrTest.h"
#include "CProcessPriorityTest.h"
#include "CProcessTest.h"
#include "CProgNameTest.h"
#include "CProgramCountersTest.h"
#include "CRapidJsonLineWriterTest.h"
#include "CRapidJsonWriterBaseTest.h"
#include "CRapidXmlParserTest.h"
#include "CRapidXmlStatePersistInserterTest.h"
#include "CRapidXmlStateRestoreTraverserTest.h"
#include "CReadWriteLockTest.h"
#include "CRegexFilterTest.h"
#include "CRegexTest.h"
#include "CResourceLocatorTest.h"
#include "CShellArgQuoterTest.h"
#include "CSleepTest.h"
#include "CSmallVectorTest.h"
#include "CStateCompressorTest.h"
#include "CStateMachineTest.h"
#include "CStaticThreadPoolTest.h"
#include "CStopWatchTest.h"
#include "CStoredStringPtrTest.h"
#include "CStringSimilarityTesterTest.h"
#include "CStringUtilsTest.h"
#include "CThreadFarmTest.h"
#include "CThreadMutexConditionTest.h"
#include "CTickerTest.h"
#include "CTimeUtilsTest.h"
#include "CTripleTest.h"
#include "CUnameTest.h"
#include "CVectorRangeTest.h"
#include "CWindowsErrorTest.h"
#include "CWordDictionaryTest.h"
#include "CWordExtractorTest.h"
#include "CXmlNodeWithChildrenTest.h"
#include "CXmlParserTest.h"

int main(int argc, const char** argv) {
    ml::test::CTestRunner runner(argc, argv);

    runner.addTest(CAllocationStrategyTest::suite());
    runner.addTest(CBase64FilterTest::suite());
    runner.addTest(CBlockingMessageQueueTest::suite());
    runner.addTest(CByteSwapperTest::suite());
    runner.addTest(CCompressedDictionaryTest::suite());
    runner.addTest(CCompressUtilsTest::suite());
    runner.addTest(CConcurrencyTest::suite());
    runner.addTest(CConcurrentWrapperTest::suite());
    runner.addTest(CContainerPrinterTest::suite());
    runner.addTest(CContainerThroughputTest::suite());
    runner.addTest(CDataFrameTest::suite());
    runner.addTest(CDelimiterTest::suite());
    runner.addTest(CDetachedProcessSpawnerTest::suite());
    runner.addTest(CDualThreadStreamBufTest::suite());
    runner.addTest(CFileDeleterTest::suite());
    runner.addTest(CFlatPrefixTreeTest::suite());
    runner.addTest(CFunctionalTest::suite());
    runner.addTest(CHashingTest::suite());
    runner.addTest(CHexUtilsTest::suite());
    runner.addTest(CIEEE754Test::suite());
    runner.addTest(CJsonLogLayoutTest::suite());
    runner.addTest(CJsonOutputStreamWrapperTest::suite());
    runner.addTest(CJsonStatePersistInserterTest::suite());
    runner.addTest(CJsonStateRestoreTraverserTest::suite());
    runner.addTest(CLoggerTest::suite());
    runner.addTest(CLoopProgressTest::suite());
    runner.addTest(CMapPopulationTest::suite());
    runner.addTest(CMemoryUsageJsonWriterTest::suite());
    runner.addTest(CMemoryUsageTest::suite());
    runner.addTest(CMessageBufferTest::suite());
    runner.addTest(CMessageQueueTest::suite());
    runner.addTest(CMonotonicTimeTest::suite());
    runner.addTest(CMutexTest::suite());
    runner.addTest(CNamedPipeFactoryTest::suite());
    runner.addTest(COsFileFuncsTest::suite());
    runner.addTest(CPatternSetTest::suite());
    runner.addTest(CPersistUtilsTest::suite());
    runner.addTest(CPolymorphicStackObjectCPtrTest::suite());
    runner.addTest(CProcessTest::suite());
    runner.addTest(CProcessPriorityTest::suite());
    runner.addTest(CProgNameTest::suite());
    runner.addTest(CProgramCountersTest::suite());
    runner.addTest(CRapidJsonLineWriterTest::suite());
    runner.addTest(CRapidJsonWriterBaseTest::suite());
    runner.addTest(CRapidXmlParserTest::suite());
    runner.addTest(CRapidXmlStatePersistInserterTest::suite());
    runner.addTest(CRapidXmlStateRestoreTraverserTest::suite());
    runner.addTest(CReadWriteLockTest::suite());
    runner.addTest(CRegexFilterTest::suite());
    runner.addTest(CRegexTest::suite());
    runner.addTest(CResourceLocatorTest::suite());
    runner.addTest(CShellArgQuoterTest::suite());
    runner.addTest(CSleepTest::suite());
    runner.addTest(CSmallVectorTest::suite());
    runner.addTest(CStateCompressorTest::suite());
    runner.addTest(CStateMachineTest::suite());
    runner.addTest(CStaticThreadPoolTest::suite());
    runner.addTest(CStopWatchTest::suite());
    runner.addTest(CStoredStringPtrTest::suite());
    runner.addTest(CStringSimilarityTesterTest::suite());
    runner.addTest(CStringUtilsTest::suite());
    runner.addTest(CThreadFarmTest::suite());
    runner.addTest(CThreadMutexConditionTest::suite());
    runner.addTest(CTickerTest::suite());
    runner.addTest(CTimeUtilsTest::suite());
    runner.addTest(CTripleTest::suite());
    runner.addTest(CUnameTest::suite());
    runner.addTest(CVectorRangeTest::suite());
    runner.addTest(CWindowsErrorTest::suite());
    runner.addTest(CWordDictionaryTest::suite());
    runner.addTest(CWordExtractorTest::suite());
    runner.addTest(CXmlNodeWithChildrenTest::suite());
    runner.addTest(CXmlParserTest::suite());

    return !runner.runTests();
}
