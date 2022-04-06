cmake_minimum_required(VERSION 3.7)

project("ML Core lib unit tests")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../../cmake/Modules/")
set(Boost_COMPILER -clang-darwin13)
set (CMAKE_CXX_STANDARD 17)

find_package(Boost 1.77.0 REQUIRED COMPONENTS iostreams filesystem program_options regex date_time log_setup thread unit_test_framework)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

find_package(LibXml2)
if(LIBXML2_FOUND)
  include_directories(${LIBXML2_INCLUDE_DIR})
endif()

find_package(ZLIB)
if(ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIR})
endif()

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
