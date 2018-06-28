#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=$(OBJS_DIR)/libMlCore$(DYNAMIC_LIB_EXT)

USE_BOOST=1
USE_BOOST_REGEX_LIBS=1
USE_BOOST_DATETIME_LIBS=1
USE_BOOST_FILESYSTEM_LIBS=1
USE_BOOST_IOSTREAMS_LIBS=1
USE_RAPIDJSON=1
USE_XML=1
USE_ZLIB=1
USE_STRPTIME=1

CPPFLAGS+= -DDYNAMIC_LIB_EXT=$(DYNAMIC_LIB_EXT)

CONF_INSTALL_DIR=$(CPP_DISTRIBUTION_HOME)/resources

# On Windows we need to copy over the Boost date/time config file that lists
# details of each timezone, whereas on Unix timezone config is part of the
# operating system.
all: build
	$(MKDIR) $(CONF_INSTALL_DIR)
	$(INSTALL) ml-en.dict $(CONF_INSTALL_DIR)
ifeq ($(OS),Windows)
	$(INSTALL) date_time_zonespec.csv $(CONF_INSTALL_DIR)
endif


PLATFORM_SRCS= \
CCondition.cc \
CCrashHandler.cc \
CCTimeR.cc \
CDetachedProcessSpawner.cc \
CFastMutex.cc \
CGmTimeR.cc \
CIEEE754.cc \
CMonotonicTime.cc \
CMutex.cc \
CNamedPipeFactory.cc \
COsFileFuncs.cc \
CPOpen.cc \
CProcess.cc \
CProcessPriority.cc \
CProgName.cc \
CReadWriteLock.cc \
CSetEnv.cc \
CSetMode.cc \
CShellArgQuoter.cc \
CSleep.cc \
CStateMachine.cc \
CStrCaseCmp.cc \
CStrFTime.cc \
CStrPTime.cc \
CStrTokR.cc \
CThread.cc \
CTimeGm.cc \
CTimezone.cc \
CUname.cc \
CUnSetEnv.cc \
CWindowsError.cc \

SRCS= \
$(OS_SRCS) \
CBase64Filter.cc \
CBufferFlushTimer.cc \
CCompressedDictionary.cc \
CCompressOStream.cc \
CompressUtils.cc \
CContainerPrinter.cc \
CDataAdder.cc \
CDataSearcher.cc \
CDelimiter.cc \
CDualThreadStreamBuf.cc \
CFileDeleter.cc \
CFlatPrefixTree.cc \
CHashing.cc \
CHexUtils.cc \
CJsonLogLayout.cc \
CJsonOutputStreamWrapper.cc \
CJsonStatePersistInserter.cc \
CJsonStateRestoreTraverser.cc \
CLogger.cc \
CMemory.cc \
CMemoryUsage.cc \
CMemoryUsageJsonWriter.cc \
CPatternSet.cc \
CPersistUtils.cc \
CRapidJsonConcurrentLineWriter.cc \
CRapidXmlParser.cc \
CRapidXmlStatePersistInserter.cc \
CRapidXmlStateRestoreTraverser.cc \
CRegex.cc \
CRegexFilter.cc \
CResourceLocator.cc \
CScopedFastLock.cc \
CScopedLock.cc \
CScopedReadLock.cc \
CScopedWriteLock.cc \
CStat.cc \
CStateCompressor.cc \
CStateDecompressor.cc \
CStatePersistInserter.cc \
CStateRestoreTraverser.cc \
CStatistics.cc \
CStopWatch.cc \
CStreamUtils.cc \
CStoredStringPtr.cc \
CStringCache.cc \
CStringSimilarityTester.cc \
CStringUtils.cc \
CTimeUtils.cc \
CWordDictionary.cc \
CWordExtractor.cc \
CXmlNode.cc \
CXmlNodeWithChildren.cc \
CXmlNodeWithChildrenPool.cc \
CXmlParser.cc \
CXmlParserIntf.cc \

include $(CPP_SRC_HOME)/mk/dynamiclib.mk

