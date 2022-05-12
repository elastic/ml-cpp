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

project("ML Api unit tests")

include_directories(../../../include)
include_directories(SYSTEM ../../../3rd_party/include)
include_directories(SYSTEM ../../../3rd_party/eigen)
include_directories(SYSTEM ../../../3rd_party/rapidjson/include)

set (SRCS
Main.cc 
CAnomalyJobConfigTest.cc 
CAnomalyJobLimitTest.cc 
CAnomalyJobTest.cc 
CBoostedTreeInferenceModelBuilderTest.cc 
CConfigUpdaterTest.cc 
CCsvInputParserTest.cc 
CCsvOutputWriterTest.cc 
CDataFrameAnalysisInstrumentationTest.cc 
CDataFrameAnalysisRunnerTest.cc 
CDataFrameAnalysisSpecificationTest.cc 
CDataFrameAnalyzerFeatureImportanceTest.cc 
CDataFrameAnalyzerOutlierTest.cc 
CDataFrameAnalyzerTrainingTest.cc 
CDataFrameTrainBoostedTreeClassifierRunnerTest.cc 
CDataFrameTrainBoostedTreeRegressionRunnerTest.cc 
CDataFrameMockAnalysisRunner.cc 
CDetectionRulesJsonParserTest.cc 
CFieldDataCategorizerTest.cc 
CForecastRunnerTest.cc 
CGlobalCategoryIdTest.cc 
CInferenceModelMetadataTest.cc 
CIoManagerTest.cc 
CJsonOutputWriterTest.cc 
CLengthEncodedInputParserTest.cc 
CMemoryUsageEstimationResultJsonWriterTest.cc 
CMockDataAdder.cc 
CMockSearcher.cc 
CAnnotationJsonWriterTest.cc 
CModelPlotDataJsonWriterTest.cc 
CModelSnapshotJsonWriterTest.cc 
CMultiFileDataAdderTest.cc 
CNdJsonInputParserTest.cc 
CNdJsonOutputWriterTest.cc 
CNoopCategoryIdMapperTest.cc 
CPerPartitionCategoryIdMapperTest.cc 
CPersistenceManagerTest.cc 
CRestorePreviousStateTest.cc 
CResultNormalizerTest.cc 
CSingleFieldDataCategorizerTest.cc 
CSingleStreamDataAdderTest.cc 
CStateRestoreStreamFilterTest.cc 
CStringStoreTest.cc 
CTestAnomalyJob.cc 
CTestFieldDataCategorizer.cc 
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(ml_test_api ${SRCS})

target_link_libraries(ml_test_api ${Boost_LIBRARIES})
target_link_libraries(ml_test_api ${LIBXML2_LIBRARIES})
target_link_libraries(ml_test_api MlCore)
target_link_libraries(ml_test_api MlMathsCommon)
target_link_libraries(ml_test_api MlMathsAnalytics)
target_link_libraries(ml_test_api MlModel)
target_link_libraries(ml_test_api MlApi)
target_link_libraries(ml_test_api MlTest)


# we expect to be built out of source, in a sub directory
# so need to change working directory up a level to find
# test files
add_test(ml_test_api ml_test_api
)

# make "make test" run the test program
add_custom_target(test_api
     DEPENDS ml_test_api
     COMMAND ml_test_api
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
