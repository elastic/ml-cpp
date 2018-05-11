#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
include $(CPP_SRC_HOME)/mk/defines.mk

TARGET=ml_test$(EXE_EXT)

USE_BOOST=1
USE_RAPIDJSON=1
USE_EIGEN=1

LIBS:=$(LIB_ML_MATHS)

all: build

SRCS=\
	Main.cc \
	CAgglomerativeClustererTest.cc \
	CAssignmentTest.cc \
	CBasicStatisticsTest.cc \
	CBjkstUniqueValuesTest.cc \
	CBootstrapClustererTest.cc \
	CBoundingBoxTest.cc \
	CCalendarFeatureTest.cc \
	CCalendarComponentAdaptiveBucketingTest.cc \
	CCategoricalToolsTest.cc \
	CChecksumTest.cc \
	CClustererTest.cc \
	CCountMinSketchTest.cc \
	CDecayRateControllerTest.cc \
	CEntropySketchTest.cc \
	CEqualWithToleranceTest.cc \
	CForecastTest.cc \
	CGammaRateConjugateTest.cc \
	CGramSchmidtTest.cc \
	CInformationCriteriaTest.cc \
	CIntegerToolsTest.cc \
	CIntegrationTest.cc \
	CKdTreeTest.cc \
	CKMeansFastTest.cc \
	CKMeansOnlineTest.cc \
	CKMostCorrelatedTest.cc \
	CLassoLogisticRegressionTest.cc \
	CLinearAlgebraTest.cc \
	CLogNormalMeanPrecConjugateTest.cc \
	CLogTDistributionTest.cc \
	CMathsFuncsTest.cc \
	CMathsMemoryTest.cc \
	CMixtureDistributionTest.cc \
	CModelTest.cc \
	CMultimodalPriorTest.cc \
	CMultinomialConjugateTest.cc \
	CMultivariateConstantPriorTest.cc \
	CMultivariateMultimodalPriorTest.cc \
	CMultivariateNormalConjugateTest.cc \
	CMultivariateOneOfNPriorTest.cc \
	CNaiveBayesTest.cc \
	CNaturalBreaksClassifierTest.cc \
	CNormalMeanPrecConjugateTest.cc \
	COneOfNPriorTest.cc \
	COrderingsTest.cc \
	COrdinalTest.cc \
	CPackedBitVectorTest.cc \
	CPeriodicityHypothesisTestsTest.cc \
	CPoissonMeanConjugateTest.cc \
	CPriorTest.cc \
	CPRNGTest.cc \
	CProbabilityAggregatorsTest.cc \
	CProbabilityCalibratorTest.cc \
	CQDigestTest.cc \
	CQuantileSketchTest.cc \
	CRegressionTest.cc \
	CRadialBasisFunctionTest.cc \
	CRandomProjectionClustererTest.cc \
	CSamplingTest.cc \
	CSeasonalComponentTest.cc \
	CSeasonalComponentAdaptiveBucketingTest.cc \
	CSetToolsTest.cc \
	CSignalTest.cc \
	CSolversTest.cc \
	CSplineTest.cc \
	CStatisticalTestsTest.cc \
	CTimeSeriesChangeDetectorTest.cc \
	CTimeSeriesDecompositionTest.cc \
	CTimeSeriesModelTest.cc \
	CToolsTest.cc \
	CTrendComponentTest.cc \
	CTrendTestsTest.cc \
	CXMeansOnlineTest.cc \
	CXMeansOnline1dTest.cc \
	CXMeansTest.cc \
	TestUtils.cc \

include $(CPP_SRC_HOME)/mk/stdcppunit.mk
