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

project("ML Api")

set(ML_DEPENDENCIES eigen)
set(ML_LINK_LIBRARIES 
  ${Boost_LIBRARIES}
  MlCore
  MlMathsCommon
  MlMathsTimeSeries
  MlMathsAnalytics
  MlModel
  $<$<BOOL:${WIN32}>:ws2_32>
  )

ml_add_library(MlApi SHARED
  CAnnotationJsonWriter.cc
  CAnomalyJob.cc
  CAnomalyJobConfig.cc
  CAnomalyJobConfigReader.cc
  CBenchMarker.cc
  CBoostedTreeInferenceModelBuilder.cc
  CCategoryIdMapper.cc
  CCmdSkeleton.cc
  CConfigUpdater.cc
  CCsvInputParser.cc
  CCsvOutputWriter.cc
  CDataFrameAnalysisConfigReader.cc
  CDataFrameAnalysisInstrumentation.cc
  CDataFrameAnalysisRunner.cc
  CDataFrameAnalysisSpecification.cc
  CDataFrameAnalysisSpecificationJsonWriter.cc
  CDataFrameAnalyzer.cc
  CDataFrameOutliersRunner.cc
  CDataFrameTrainBoostedTreeClassifierRunner.cc
  CDataFrameTrainBoostedTreeRegressionRunner.cc
  CDataFrameTrainBoostedTreeRunner.cc
  CDataProcessor.cc
  CDetectionRulesJsonParser.cc
  CFieldDataCategorizer.cc
  CForecastRunner.cc
  CGlobalCategoryId.cc
  CHierarchicalResultsWriter.cc
  CInferenceModelDefinition.cc
  CInferenceModelMetadata.cc
  CInputParser.cc
  CIoManager.cc
  CJsonOutputWriter.cc
  CLengthEncodedInputParser.cc
  CMemoryUsageEstimationResultJsonWriter.cc
  CModelPlotDataJsonWriter.cc
  CModelSizeStatsJsonWriter.cc
  CModelSnapshotJsonWriter.cc
  CNdInputParser.cc
  CNdJsonInputParser.cc
  CNdJsonOutputWriter.cc
  CNoopCategoryIdMapper.cc
  CPerPartitionCategoryIdMapper.cc
  CPersistenceManager.cc
  CResultNormalizer.cc
  CSimpleOutputWriter.cc
  CSingleFieldDataCategorizer.cc
  CSingleStreamDataAdder.cc
  CSingleStreamSearcher.cc
  CStateRestoreStreamFilter.cc
  ElasticsearchStateIndex.cc
  )
