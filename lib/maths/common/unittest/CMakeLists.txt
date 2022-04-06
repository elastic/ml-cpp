cmake_minimum_required(VERSION 3.7)

project("ML Maths Common lib unit tests")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../../cmake/Modules/")
set(Boost_COMPILER -clang-darwin13)
set (CMAKE_CXX_STANDARD 17)

find_package(Boost 1.77.0 REQUIRED COMPONENTS iostreams filesystem program_options regex date_time log_setup thread unit_test_framework)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

add_compile_definitions(RAPIDJSON_HAS_STDSTRING DRAPIDJSON_NEON MacOSX)

include_directories(../../../../include)
include_directories(../../../../3rd_party/include)
include_directories(../../../../3rd_party/eigen)
include_directories(../../../../3rd_party/rapidjson/include)

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

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(ml_test_maths_common ${SRCS} ${PLATFORM_SRCS})

target_link_libraries(ml_test_maths_common ${Boost_LIBRARIES})
target_link_libraries(ml_test_maths_common MlCore)
target_link_libraries(ml_test_maths_common MlMathsCommon)
target_link_libraries(ml_test_maths_common MlTest)


# we expect to be built out of source, in a sub ditrectory
# so need to change working directory up a level to find
# test files
add_test(ml_test_maths_common ml_test_maths_common
)

# make "make test" run the test program
add_custom_target(test_maths_common
     DEPENDS ml_test_maths_common
     COMMAND ml_test_maths_common
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
