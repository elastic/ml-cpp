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

TARGET=$(OBJS_DIR)/libMlMaths$(DYNAMIC_LIB_EXT)

USE_BOOST=1
USE_EIGEN=1

LIBS:=$(LIB_ML_CORE)

all: build

SRCS= \
CAdaptiveBucketing.cc \
CAgglomerativeClusterer.cc \
CAssignment.cc \
CBasicStatistics.cc \
CBjkstUniqueValues.cc \
CCalendarFeature.cc \
CCalendarComponentAdaptiveBucketing.cc \
CCalendarComponent.cc \
CCategoricalTools.cc \
CClusterer.cc \
CClustererStateSerialiser.cc \
CConstantPrior.cc \
CCooccurrences.cc \
CCountMinSketch.cc \
CDecayRateController.cc \
CDecompositionComponent.cc \
CEntropySketch.cc \
CExpandingWindow.cc \
CGammaRateConjugate.cc \
CGradientDescent.cc \
CGramSchmidt.cc \
CInformationCriteria.cc \
CIntegerTools.cc \
CIntegration.cc \
CKMeansOnline1d.cc \
CKMostCorrelated.cc \
CLassoLogisticRegression.cc \
CLinearAlgebra.cc \
CLinearAlgebraTools.cc \
CLogNormalMeanPrecConjugate.cc \
CLogTDistribution.cc \
CMathsFuncs.cc \
CMixtureDistribution.cc \
CModel.cc \
CModelStateSerialiser.cc \
CModelWeight.cc \
CMultimodalPrior.cc \
CMultinomialConjugate.cc \
CMultivariateConstantPrior.cc \
CMultivariateMultimodalPrior.cc \
CMultivariateMultimodalPriorFactory.cc \
CMultivariateNormalConjugateFactory.cc \
CMultivariateOneOfNPrior.cc \
CMultivariateOneOfNPriorFactory.cc \
CMultivariatePrior.cc \
CNaturalBreaksClassifier.cc \
CNormalMeanPrecConjugate.cc \
COneOfNPrior.cc \
Constants.cc \
COrderings.cc \
COrdinal.cc \
CPackedBitVector.cc \
CPeriodicityHypothesisTests.cc \
CPoissonMeanConjugate.cc \
CPrior.cc \
CPriorStateSerialiser.cc \
CPRNG.cc \
CProbabilityCalibrator.cc \
CQDigest.cc \
CQuantileSketch.cc \
CRadialBasisFunction.cc \
CRegression.cc \
CRestoreParams.cc \
CSampling.cc \
CSeasonalComponent.cc \
CSeasonalComponentAdaptiveBucketing.cc \
CSeasonalTime.cc \
CSignal.cc \
CSpline.cc \
CStatisticalTests.cc \
CTimeSeriesDecomposition.cc \
CTimeSeriesDecompositionDetail.cc \
CTimeSeriesDecompositionStateSerialiser.cc \
CTimeSeriesDecompositionStub.cc \
CTimeSeriesModel.cc \
CTools.cc \
CTrendComponent.cc \
CTrendTests.cc \
CXMeansOnline1d.cc \
CXMeansOnlineFactory.cc \
MathsTypes.cc \
ProbabilityAggregators.cc \

include $(CPP_SRC_HOME)/mk/dynamiclib.mk

