#
# ELASTICSEARCH CONFIDENTIAL
#
# Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
#
# Notice: this software, and all information contained
# therein, is the exclusive property of Elasticsearch BV
# and its licensors, if any, and is protected under applicable
# domestic and foreign law, and international treaties.
#
# Reproduction, republication or distribution without the
# express written consent of Elasticsearch BV is
# strictly prohibited.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=ml_test$(EXE_EXT)

USE_XML=1
USE_BOOST=1
USE_BOOST_DATETIME_LIBS=1
USE_BOOST_FILESYSTEM_LIBS=1
USE_BOOST_THREAD_LIBS=1
USE_RAPIDJSON=1
USE_EIGEN=1

LIBS:=$(LIB_ML_MATHS)

all: build

PLATFORM_SRCS= \
CProcessPriorityTest.cc \

SRCS=\
$(OS_SRCS) \
Main.cc \
CAllocationStrategyTest.cc \
CBase64FilterTest.cc \
CBlockingMessageQueueTest.cc \
CByteSwapperTest.cc \
CCompressedDictionaryTest.cc \
CCompressUtilsTest.cc \
CConcurrentWrapperTest.cc \
CContainerPrinterTest.cc \
CContainerThroughputTest.cc \
CDelimiterTest.cc \
CDetachedProcessSpawnerTest.cc \
CDualThreadStreamBufTest.cc \
CFileDeleterTest.cc \
CFlatPrefixTreeTest.cc \
CFunctionalTest.cc \
CHashingTest.cc \
CHexUtilsTest.cc \
CIEEE754Test.cc \
CJsonLogLayoutTest.cc \
CJsonOutputStreamWrapperTest.cc \
CJsonStatePersistInserterTest.cc \
CJsonStateRestoreTraverserTest.cc \
CLoggerTest.cc \
CMemoryUsageJsonWriterTest.cc \
CMemoryUsageTest.cc \
CMessageBufferTest.cc \
CMessageQueueTest.cc \
CMonotonicTimeTest.cc \
CMapPopulationTest.cc \
CMutexTest.cc \
CNamedPipeFactoryTest.cc \
COsFileFuncsTest.cc \
CPatternSetTest.cc \
CPersistUtilsTest.cc \
CPolymorphicStackObjectCPtrTest.cc \
CProcessTest.cc \
CProgNameTest.cc \
CRapidJsonLineWriterTest.cc\
CRapidJsonWriterBaseTest.cc\
CRapidXmlParserTest.cc \
CRapidXmlStatePersistInserterTest.cc \
CRapidXmlStateRestoreTraverserTest.cc \
CReadWriteLockTest.cc \
CRegexFilterTest.cc \
CRegexTest.cc \
CResourceLocatorTest.cc \
CShellArgQuoterTest.cc \
CSleepTest.cc \
CSmallVectorTest.cc \
CStateCompressorTest.cc \
CStateMachineTest.cc \
CStatisticsTest.cc \
CStopWatchTest.cc \
CStoredStringPtrTest.cc \
CStringSimilarityTesterTest.cc \
CStringUtilsTest.cc \
CThreadFarmTest.cc \
CThreadPoolTest.cc \
CThreadMutexConditionTest.cc \
CTickerTest.cc \
CTimeUtilsTest.cc \
CTripleTest.cc \
CUnameTest.cc \
CVectorRangeTest.cc \
CWindowsErrorTest.cc \
CWordDictionaryTest.cc \
CWordExtractorTest.cc \
CXmlNodeWithChildrenTest.cc \
CXmlParserTest.cc \

include $(CPP_SRC_HOME)/mk/stdcppunit.mk

