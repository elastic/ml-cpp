#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=ml_test$(EXE_EXT)

USE_XML=1
USE_BOOST=1
USE_BOOST_DATETIME_LIBS=1
USE_BOOST_FILESYSTEM_LIBS=1
USE_BOOST_THREAD_LIBS=1
USE_BOOST_TEST_LIBS=1
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
CCompressedDictionaryTest.cc \
CCompressUtilsTest.cc \
CConcurrencyTest.cc \
CConcurrentWrapperTest.cc \
CContainerPrinterTest.cc \
CContainerThroughputTest.cc \
CCsvLineParserTest.cc \
CDataFrameTest.cc \
CDetachedProcessSpawnerTest.cc \
CDualThreadStreamBufTest.cc \
CFlatPrefixTreeTest.cc \
CFunctionalTest.cc \
CHashingTest.cc \
CIEEE754Test.cc \
CImmutableRadixSetTest.cc \
CJsonLogLayoutTest.cc \
CJsonOutputStreamWrapperTest.cc \
CJsonStatePersistInserterTest.cc \
CJsonStateRestoreTraverserTest.cc \
CLoggerTest.cc \
CLoopProgressTest.cc \
CMemoryUsageJsonWriterTest.cc \
CMemoryUsageTest.cc \
CMonotonicTimeTest.cc \
CMutexTest.cc \
CNamedPipeFactoryTest.cc \
COsFileFuncsTest.cc \
CPackedBitVectorTest.cc \
CPatternSetTest.cc \
CPersistenceTagTest.cc \
CPersistUtilsTest.cc \
CProcessTest.cc \
CProgNameTest.cc \
CProgramCountersTest.cc \
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
CStaticThreadPoolTest.cc \
CStopWatchTest.cc \
CStoredStringPtrTest.cc \
CStringSimilarityTesterTest.cc \
CStringUtilsTest.cc \
CThreadMutexConditionTest.cc \
CTimeUtilsTest.cc \
CTripleTest.cc \
CUnameTest.cc \
CVectorRangeTest.cc \
CWindowsErrorTest.cc \
CWordDictionaryTest.cc \
CWordExtractorTest.cc \
CXmlNodeWithChildrenTest.cc \
CXmlParserTest.cc \

include $(CPP_SRC_HOME)/mk/stdboosttest.mk

