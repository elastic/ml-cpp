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

TARGET=$(OBJS_DIR)/libMlModel$(DYNAMIC_LIB_EXT)

USE_BOOST=1
USE_BOOST_FILESYSTEM_LIBS=1
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
CForecastModelPersist.cc \
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
