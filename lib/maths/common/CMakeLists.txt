cmake_minimum_required(VERSION 3.7)
project("ML Maths Common lib")

set (CMAKE_CXX_STANDARD 17)

set(Boost_COMPILER -clang-darwin13)
find_package(Boost 1.77.0 REQUIRED COMPONENTS iostreams filesystem program_options regex date_time log_setup thread unit_test_framework)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

add_compile_definitions(RAPIDJSON_HAS_STDSTRING DRAPIDJSON_NEON MacOSX)

include_directories(../../../include)
include_directories(../../../3rd_party/include)
include_directories(../../../3rd_party/eigen)
include_directories(../../../3rd_party/rapidjson/include)

set(SRCS
CAgglomerativeClusterer.cc 
CBasicStatistics.cc 
CBayesianOptimisation.cc 
CBjkstUniqueValues.cc 
CCategoricalTools.cc 
CChecksum.cc 
CClusterer.cc 
CClustererStateSerialiser.cc 
CConstantPrior.cc 
CEntropySketch.cc 
CFuzzyLogic.cc 
CGammaRateConjugate.cc 
CGradientDescent.cc 
CInformationCriteria.cc 
CIntegerTools.cc 
CIntegration.cc 
CKMeansOnline1d.cc 
CKMostCorrelated.cc 
CLeastSquaresOnlineRegression.cc 
CLinearAlgebra.cc 
CLinearAlgebraTools.cc 
CLogNormalMeanPrecConjugate.cc 
CLogTDistribution.cc 
CMathsFuncs.cc 
CMixtureDistribution.cc 
CModel.cc 
CModelWeight.cc 
CMultimodalPrior.cc 
CMultinomialConjugate.cc 
CMultivariateConstantPrior.cc 
CMultivariateMultimodalPrior.cc 
CMultivariateMultimodalPriorFactory.cc 
CMultivariateNormalConjugateFactory.cc 
CMultivariateOneOfNPrior.cc 
CMultivariateOneOfNPriorFactory.cc 
CMultivariatePrior.cc 
CNaiveBayes.cc 
CNaturalBreaksClassifier.cc 
CNormalMeanPrecConjugate.cc 
COneOfNPrior.cc 
COrderings.cc 
CPRNG.cc 
CPoissonMeanConjugate.cc 
CPrior.cc 
CPriorStateSerialiser.cc 
CQDigest.cc 
CQuantileSketch.cc 
CRestoreParams.cc 
CSampling.cc 
CSpline.cc 
CStatisticalTests.cc 
CTools.cc 
CXMeansOnline1d.cc 
CXMeansOnlineFactory.cc 
MathsTypes.cc 
ProbabilityAggregators.cc 
)

add_library(MlMathsCommon SHARED ${SRCS})
target_link_libraries(MlMathsCommon ${Boost_LIBRARIES})
target_link_libraries(MlMathsCommon MlCore)

install(TARGETS MlMathsCommon DESTINATION lib)
