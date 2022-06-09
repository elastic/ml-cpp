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

project("ML Maths Common unit tests")

set (SRCS
  Main.cc
  CAgglomerativeClustererTest.cc
  CBasicStatisticsTest.cc
  CBayesianOptimisationTest.cc
  CBjkstUniqueValuesTest.cc
  CBootstrapClustererTest.cc
  CBoundingBoxTest.cc
  CCategoricalToolsTest.cc
  CChecksumTest.cc
  CClustererTest.cc
  CEntropySketchTest.cc
  CEqualWithToleranceTest.cc
  CGammaRateConjugateTest.cc
  CInformationCriteriaTest.cc
  CIntegerToolsTest.cc
  CIntegrationTest.cc
  CKMeansOnlineTest.cc
  CKMeansTest.cc
  CKMostCorrelatedTest.cc
  CKdTreeTest.cc
  CLbfgsTest.cc
  CLeastSquaresOnlineRegressionTest.cc
  CLinearAlgebraTest.cc
  CLogNormalMeanPrecConjugateTest.cc
  CLogTDistributionTest.cc
  CLowessTest.cc
  CMathsFuncsTest.cc
  CMixtureDistributionTest.cc
  CModelTest.cc
  CMultimodalPriorTest.cc
  CMultinomialConjugateTest.cc
  CMultivariateConstantPriorTest.cc
  CMultivariateMultimodalPriorTest.cc
  CMultivariateNormalConjugateTest.cc
  CMultivariateOneOfNPriorTest.cc
  CNaiveBayesTest.cc
  CNaturalBreaksClassifierTest.cc
  CNormalMeanPrecConjugateTest.cc
  COneOfNPriorTest.cc
  COrderingsTest.cc
  COrthogonaliserTest.cc
  CPRNGTest.cc
  CPoissonMeanConjugateTest.cc
  CPriorTest.cc
  CProbabilityAggregatorsTest.cc
  CQDigestTest.cc
  CQuantileSketchTest.cc
  CRandomProjectionClustererTest.cc
  CSamplingTest.cc
  CSetToolsTest.cc
  CSolversTest.cc
  CSplineTest.cc
  CStatisticalTestsTest.cc
  CToolsTest.cc
  CXMeansOnline1dTest.cc
  CXMeansOnlineTest.cc
  CXMeansTest.cc
  TestUtils.cc
  )

set(ML_LINK_LIBRARIES
  ${Boost_LIBRARIES_WITH_UNIT_TEST}
  MlCore
  MlMathsCommon
  MlTest
  )

ml_add_test_executable(maths_common ${SRCS})
