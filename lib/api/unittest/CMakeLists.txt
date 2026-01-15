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

set (SRCS
  Main.cc
  CAnnotationJsonWriterTest.cc
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
  CDataFrameMockAnalysisRunner.cc
  CDataFrameTrainBoostedTreeClassifierRunnerTest.cc
  CDataFrameTrainBoostedTreeRegressionRunnerTest.cc
  CDataSummarizationJsonSerializerTest.cc
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
  CModelPlotDataJsonWriterTest.cc
  CModelSnapshotJsonWriterTest.cc
  CMultiFileDataAdderTest.cc
  CNdJsonInputParserTest.cc
  CNdJsonOutputWriterTest.cc
  CNoopCategoryIdMapperTest.cc
  CPerPartitionCategoryIdMapperTest.cc
  CPersistenceManagerTest.cc
  CResultNormalizerTest.cc
  CSerializableToJsonTest.cc
  CSingleFieldDataCategorizerTest.cc
  CSingleStreamDataAdderTest.cc
  CStateRestoreStreamFilterTest.cc
  CTestAnomalyJob.cc
  CTestFieldDataCategorizer.cc
  MyForecastTest.cc
  )

set(ML_LINK_LIBRARIES
  ${Boost_LIBRARIES_WITH_UNIT_TEST}
  ${LIBXML2_LIBRARIES}
  MlCore
  MlMathsCommon
  MlMathsAnalytics
  MlModel
  MlApi
  MlTest
  )

# Valijson must only be used in test cases, never in production code
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/3rd_party/valijson/include)

ml_add_test_executable(api ${SRCS})
