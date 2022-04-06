cmake_minimum_required(VERSION 3.7)

project("ML Api lib unit tests")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../../cmake/Modules/")
set(Boost_COMPILER -clang-darwin13)
set (CMAKE_CXX_STANDARD 17)

find_package(Boost 1.77.0 REQUIRED COMPONENTS iostreams filesystem program_options regex date_time log_setup thread unit_test_framework)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

#RAPIDJSONINCLUDES=-isystem $(CPP_SRC_HOME)/3rd_party/rapidjson/include
#ifeq ($(HARDWARE_ARCH),x86_64)
#RAPIDJSONCPPFLAGS=-DRAPIDJSON_HAS_STDSTRING -DRAPIDJSON_SSE42
#else
#RAPIDJSONCPPFLAGS=-DRAPIDJSON_HAS_STDSTRING -DRAPIDJSON_NEON
#endif

#if (MSVC)
#    # warning level 4 and all warnings as errors
#    add_compile_options(/W4 /WX)
#else()
#    # lots of warnings and all warnings as errors
#    add_compile_options(-Wall -Wextra -pedantic -Werror)
#endif()

#add_compile_definitions(-DRAPIDJSON_HAS_STDSTRING -DRAPIDJSON_NEON)
add_compile_definitions(RAPIDJSON_HAS_STDSTRING DRAPIDJSON_NEON MacOSX)

include_directories(../../../include)
include_directories(../../../3rd_party/include)
include_directories(../../../3rd_party/eigen)
include_directories(../../../3rd_party/rapidjson/include)

# TODO make platform safe
set (PLATFORM_SRCS
)

set (SRCS
Main.cc 
CAnomalyJobConfigTest.cc 
CAnomalyJobLimitTest.cc 
CAnomalyJobTest.cc 
CBoostedTreeInferenceModelBuilderTest.cc 
CConfigUpdaterTest.cc 
CCsvInputParserTest.cc 
CCsvOutputWriterTest.cc 
CDataFrameAnalysisInstrumentationTest.cc 
CDataFrameAnalysisRunnerTest.cc 
CDataFrameAnalysisSpecificationTest.cc 
CDataFrameAnalyzerFeatureImportanceTest.cc 
CDataFrameAnalyzerOutlierTest.cc 
CDataFrameAnalyzerTrainingTest.cc 
CDataFrameTrainBoostedTreeClassifierRunnerTest.cc 
CDataFrameTrainBoostedTreeRegressionRunnerTest.cc 
CDataFrameMockAnalysisRunner.cc 
CDetectionRulesJsonParserTest.cc 
CFieldDataCategorizerTest.cc 
CForecastRunnerTest.cc 
CGlobalCategoryIdTest.cc 
CInferenceModelMetadataTest.cc 
CIoManagerTest.cc 
CJsonOutputWriterTest.cc 
CLengthEncodedInputParserTest.cc 
CMemoryUsageEstimationResultJsonWriterTest.cc 
CMockDataAdder.cc 
CMockSearcher.cc 
CAnnotationJsonWriterTest.cc 
CModelPlotDataJsonWriterTest.cc 
CModelSnapshotJsonWriterTest.cc 
CMultiFileDataAdderTest.cc 
CNdJsonInputParserTest.cc 
CNdJsonOutputWriterTest.cc 
CNoopCategoryIdMapperTest.cc 
CPerPartitionCategoryIdMapperTest.cc 
CPersistenceManagerTest.cc 
CRestorePreviousStateTest.cc 
CResultNormalizerTest.cc 
CSingleFieldDataCategorizerTest.cc 
CSingleStreamDataAdderTest.cc 
CStateRestoreStreamFilterTest.cc 
CStringStoreTest.cc 
CTestAnomalyJob.cc 
CTestFieldDataCategorizer.cc 
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(ml_test_api ${SRCS} ${PLATFORM_SRCS})

#TODO make portable for windows
set(NETLIBS "")

target_link_libraries(ml_test_api ${Boost_LIBRARIES})
target_link_libraries(ml_test_api ${LIBXML2_LIBRARIES})
target_link_libraries(ml_test_api MlCore)
target_link_libraries(ml_test_api MlMathsCommon)
target_link_libraries(ml_test_api MlMathsAnalytics)
target_link_libraries(ml_test_api MlModel)
target_link_libraries(ml_test_api MlApi)
target_link_libraries(ml_test_api MlTest)
target_link_libraries(ml_test_api ${NETLIBS})


# we expect to be built out of source, in a sub ditrectory
# so need to change working directory up a level to find
# test files
add_test(ml_test_api ml_test_api
)

# make "make test" run the test program
add_custom_target(test_api
     DEPENDS ml_test_api
     COMMAND ml_test_api
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
