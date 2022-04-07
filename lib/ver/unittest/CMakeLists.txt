cmake_minimum_required(VERSION 3.7)

project("ML Core lib unit tests")

include_directories(../../../include)

# TODO make platform safe
set (PLATFORM_SRCS
)

set (SRCS
Main.cc 
CBuildInfoTest.cc
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(ml_test_ver ${SRCS} ${PLATFORM_SRCS})

target_link_libraries(ml_test_ver ${Boost_LIBRARIES})
target_link_libraries(ml_test_ver ${LIBXML2_LIBRARIES})
target_link_libraries(ml_test_ver MlCore)
target_link_libraries(ml_test_ver MlVer)
target_link_libraries(ml_test_ver MlTest)

add_test(ml_test_ver ml_test_ver)
# make "make test" run the test program
add_custom_target(test_ver
     DEPENDS ml_test_ver
     COMMAND ml_test_ver
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
