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

project("ML Maths Common")

include_directories(../../../include)
include_directories(SYSTEM ../../../3rd_party/include)
include_directories(SYSTEM ../../../3rd_party/eigen)
include_directories(SYSTEM ../../../3rd_party/rapidjson/include)

add_compile_definitions(BUILDING_libMlMathsCommon)

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

add_dependencies(MlMathsCommon eigen)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(MlMathsCommon PRIVATE "-current_version ${ML_VERSION_NUM}" "-compatibility_version ${ML_VERSION_NUM}" "${COVERAGE}")
endif()

install(TARGETS MlMathsCommon RUNTIME DESTINATION ${DYNAMIC_LIB_DIR} ARCHIVE DESTINATION ${IMPORT_LIB_DIR})
if(MSVC)
  set_property(TARGET MlMathsCommon PROPERTY COMPILE_PDB_NAME MlMathsCommon)
  install(FILES $<TARGET_PDB_FILE:MlMathsCommon> DESTINATION ${DYNAMIC_LIB_DIR} OPTIONAL)
endif()
