cmake_minimum_required(VERSION 3.7)

project("ML Maths Analytics lib unit tests")

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
CBoostedTreeLeafNodeStatisticsTest.cc 
CBoostedTreeLossTest.cc 
CBoostedTreeTest.cc 
CDataFrameCategoryEncoderTest.cc 
CDataFrameUtilsTest.cc 
CMicTest.cc 
COutliersTest.cc 
CTreeShapFeatureImportanceTest.cc 
TestUtils.cc 
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(ml_test_maths_analytics ${SRCS} ${PLATFORM_SRCS})

target_link_libraries(ml_test_maths_analytics ${Boost_LIBRARIES})
target_link_libraries(ml_test_maths_analytics MlCore)
target_link_libraries(ml_test_maths_analytics MlMathsCommon)
target_link_libraries(ml_test_maths_analytics MlMathsAnalytics)
target_link_libraries(ml_test_maths_analytics MlTest)


# we expect to be built out of source, in a sub ditrectory
# so need to change working directory up a level to find
# test files
add_test(ml_test_maths_analytics ml_test_maths_analytics
)

# make "make test" run the test program
add_custom_target(test_maths_analytics
     DEPENDS ml_test_maths_analytics
     COMMAND ml_test_maths_analytics
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
