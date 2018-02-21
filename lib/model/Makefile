#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=$(OBJS_DIR)/libMlModel$(DYNAMIC_LIB_EXT)

USE_BOOST=1
USE_RAPIDJSON=1
USE_EIGEN=1

LIBS:=$(LIB_ML_CORE) $(LIB_ML_MATHS)

all: build

SRCS= \
CAnnotatedProbability.cc \
CAnnotatedProbabilityBuilder.cc \
CAnomalyDetector.cc \
CAnomalyDetectorModel.cc \
CAnomalyDetectorModelConfig.cc \
CAnomalyScore.cc \
CBucketGatherer.cc \
CCountingModel.cc \
CCountingModelFactory.cc \
CDataClassifier.cc \
CDataGatherer.cc \
CDetectionRule.cc \
CDetectorEqualizer.cc \
CDynamicStringIdRegistry.cc \
CEventData.cc \
CEventRateBucketGatherer.cc \
CEventRateModel.cc \
CEventRateModelFactory.cc \
CEventRatePopulationModel.cc \
CEventRatePopulationModelFactory.cc \
CFeatureData.cc \
CForecastDataSink.cc \
CGathererTools.cc \
CHierarchicalResults.cc \
CHierarchicalResultsAggregator.cc \
CHierarchicalResultsNormalizer.cc \
CHierarchicalResultsPopulator.cc \
CHierarchicalResultsProbabilityFinalizer.cc \
CIndividualModel.cc \
CInterimBucketCorrector.cc \
CLimits.cc \
CMemoryUsageEstimator.cc \
CMetricBucketGatherer.cc \
CMetricModel.cc \
CMetricModelFactory.cc \
CMetricPopulationModel.cc \
CMetricPopulationModelFactory.cc \
CModelDetailsView.cc \
CModelFactory.cc \
CModelParams.cc \
CModelPlotData.cc \
CModelTools.cc \
CPartitioningFields.cc \
CPopulationModel.cc \
CProbabilityAndInfluenceCalculator.cc \
CResourceMonitor.cc \
CResultsQueue.cc \
CRuleCondition.cc \
CSample.cc \
CSampleCounts.cc \
CSearchKey.cc \
CSimpleCountDetector.cc \
CStringStore.cc \
FrequencyPredicates.cc \
FunctionTypes.cc \
ModelTypes.cc

include $(CPP_SRC_HOME)/mk/dynamiclib.mk
