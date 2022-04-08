project("ML Maths Common lib")

include_directories(../../../include)
include_directories(SYSTEM ../../../3rd_party/include)
include_directories(SYSTEM ../../../3rd_party/eigen)
include_directories(SYSTEM ../../../3rd_party/rapidjson/include)

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
target_link_libraries(MlMathsCommon PUBLIC ${Boost_LIBRARIES})
target_link_libraries(MlMathsCommon PUBLIC MlCore)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(MlMathsCommon PRIVATE "-current_version ${ML_VERSION_NUM}" "-compatibility_version ${ML_VERSION_NUM}" "${COVERAGE}")
endif()

install(TARGETS MlMathsCommon DESTINATION lib)
