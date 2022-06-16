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

ml_add_test_executable(api ${SRCS})
