project("ML Core lib unit tests")

include_directories(../../../include)
include_directories(SYSTEM ../../../3rd_party/include)
include_directories(SYSTEM ../../../3rd_party/eigen)
include_directories(SYSTEM ../../../3rd_party/rapidjson/include)

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
