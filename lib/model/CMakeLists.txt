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

project("ML Model")

set(ML_DEPENDENCIES third_party)
set(ML_LINK_LIBRARIES 
  ${Boost_LIBRARIES}
  MlCore
  MlMathsCommon
  MlMathsTimeSeries)

ml_add_library(MlModel SHARED
  CAnnotatedProbability.cc
  CAnnotatedProbabilityBuilder.cc
  CAnomalyDetector.cc
  CAnomalyDetectorModel.cc
  CAnomalyDetectorModelConfig.cc
  CAnomalyScore.cc
  CBucketGatherer.cc
  CCategoryExamplesCollector.cc
  CCountingModel.cc
  CCountingModelFactory.cc
  CDataCategorizer.cc
  CDataClassifier.cc
  CDataGatherer.cc
  CDetectionRule.cc
  CDetectorEqualizer.cc
  CDynamicStringIdRegistry.cc
  CEventData.cc
  CEventRateBucketGatherer.cc
  CEventRateModel.cc
  CEventRateModelFactory.cc
  CEventRatePopulationModel.cc
  CEventRatePopulationModelFactory.cc
  CFeatureData.cc
  CForecastDataSink.cc
  CForecastModelPersist.cc
  CGathererTools.cc
  CHierarchicalResults.cc
  CHierarchicalResultsAggregator.cc
  CHierarchicalResultsNormalizer.cc
  CHierarchicalResultsPopulator.cc
  CHierarchicalResultsProbabilityFinalizer.cc
  CIndividualModel.cc
  CInterimBucketCorrector.cc
  CLimits.cc
  CLocalCategoryId.cc
  CMemoryUsageEstimator.cc
  CMetricBucketGatherer.cc
  CMetricModel.cc
  CMetricModelFactory.cc
  CMetricPopulationModel.cc
  CMetricPopulationModelFactory.cc
  CAnnotation.cc
  CModelDetailsView.cc
  CModelFactory.cc
  CModelPlotData.cc
  CModelTools.cc
  CMonitoredResource.cc
  CPartitioningFields.cc
  CPopulationModel.cc
  CProbabilityAndInfluenceCalculator.cc
  CResourceMonitor.cc
  CRuleCondition.cc
  CRuleScope.cc
  CSample.cc
  CSampleCounts.cc
  CSearchKey.cc
  CSimpleCountDetector.cc
  CStringStore.cc
  CTokenListCategory.cc
  CTokenListDataCategorizerBase.cc
  CTokenListReverseSearchCreator.cc
  FrequencyPredicates.cc
  FunctionTypes.cc
  ModelTypes.cc
  SModelParams.cc
  )
