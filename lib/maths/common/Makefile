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
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=$(OBJS_DIR)/libMlMathsCommon$(DYNAMIC_LIB_EXT)

USE_BOOST=1
USE_BOOST_FILESYSTEM_LIBS=1
USE_RAPIDJSON=1
USE_EIGEN=1

LIBS:=$(LIB_ML_CORE)

all: build

SRCS= \
CAgglomerativeClusterer.cc \
CBasicStatistics.cc \
CBayesianOptimisation.cc \
CBjkstUniqueValues.cc \
CCategoricalTools.cc \
CChecksum.cc \
CClusterer.cc \
CClustererStateSerialiser.cc \
CConstantPrior.cc \
CEntropySketch.cc \
CFuzzyLogic.cc \
CGammaRateConjugate.cc \
CGradientDescent.cc \
CInformationCriteria.cc \
CIntegerTools.cc \
CIntegration.cc \
CKMeansOnline1d.cc \
CKMostCorrelated.cc \
CLeastSquaresOnlineRegression.cc \
CLinearAlgebra.cc \
CLinearAlgebraTools.cc \
CLogNormalMeanPrecConjugate.cc \
CLogTDistribution.cc \
CMathsFuncs.cc \
CMixtureDistribution.cc \
CModel.cc \
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
CNaiveBayes.cc \
CNaturalBreaksClassifier.cc \
CNormalMeanPrecConjugate.cc \
COneOfNPrior.cc \
COrderings.cc \
CPRNG.cc \
CPoissonMeanConjugate.cc \
CPrior.cc \
CPriorStateSerialiser.cc \
CQDigest.cc \
CQuantileSketch.cc \
CRestoreParams.cc \
CSampling.cc \
CSpline.cc \
CStatisticalTests.cc \
CTools.cc \
CXMeansOnline1d.cc \
CXMeansOnlineFactory.cc \
MathsTypes.cc \
ProbabilityAggregators.cc \

include $(CPP_SRC_HOME)/mk/dynamiclib.mk

