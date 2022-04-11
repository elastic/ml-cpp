cmake_minimum_required(VERSION 3.7)

project("ML controller unit tests")

include_directories(../../../include)
include_directories(SYSTEM ../../../3rd_party/include)
include_directories(SYSTEM ../../../3rd_party/eigen)
include_directories(SYSTEM ../../../3rd_party/rapidjson/include)

# TODO make platform safe
set (PLATFORM_SRCS
)

set (SRCS
    Main.cc
    CBlockingCallCancellingStreamMonitorTest.cc
    CCommandProcessorTest.cc
    CResponseJsonWriterTest.cc
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(ml_test_controller ${SRCS} ${PLATFORM_SRCS})

target_link_libraries(ml_test_controller ${Boost_LIBRARIES})
target_link_libraries(ml_test_controller ${LIBXML2_LIBRARIES})
target_link_libraries(ml_test_controller MlCore)
target_link_libraries(ml_test_controller MlTest)
target_link_libraries(ml_test_controller MlVer)

file(GLOB_RECURSE CONTROLLER_OBJS ${CMAKE_CURRENT_BINARY_DIR}/../C*.cc.o)
list(FILTER CONTROLLER_OBJS EXCLUDE REGEX "Test" )
target_link_libraries(ml_test_controller ${CONTROLLER_OBJS})

# we expect to be built out of source, in a sub ditrectory
# so need to change working directory up a level to find
# test files
add_test(ml_test_controller ml_test_controller
)

# make "make test" run the test program
add_custom_target(test_controller
     DEPENDS ml_test_controller
     COMMAND ml_test_controller
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
