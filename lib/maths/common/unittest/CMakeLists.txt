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

include_directories(../../../../include)
include_directories(SYSTEM ../../../../3rd_party/include)
include_directories(SYSTEM ../../../../3rd_party/eigen)
include_directories(SYSTEM ../../../../3rd_party/rapidjson/include)

# TODO make platform safe
set (PLATFORM_SRCS
)

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

add_executable(ml_test_maths_common ${SRCS} ${PLATFORM_SRCS})

target_link_libraries(ml_test_maths_common ${Boost_LIBRARIES})
target_link_libraries(ml_test_maths_common libMlCore)
target_link_libraries(ml_test_maths_common libMlMathsCommon)
target_link_libraries(ml_test_maths_common libMlTest)


add_test(ml_test_maths_common ml_test_maths_common
)

# make "make test" run the test program
add_custom_target(test_maths_common
     DEPENDS ml_test_maths_common
     COMMAND ml_test_maths_common
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
