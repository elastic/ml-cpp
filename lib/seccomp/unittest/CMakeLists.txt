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
CSystemCallFilterTest.cc 
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(ml_test_seccomp ${SRCS} ${PLATFORM_SRCS})

target_link_libraries(ml_test_seccomp ${Boost_LIBRARIES})
target_link_libraries(ml_test_seccomp ${LIBXML2_LIBRARIES})
target_link_libraries(ml_test_seccomp MlCore)
target_link_libraries(ml_test_seccomp MlSeccomp)
target_link_libraries(ml_test_seccomp MlTest)

add_test(ml_test_seccomp ml_test_seccomp)
# make "make test" run the test program
add_custom_target(test_seccomp
     DEPENDS ml_test_seccomp
     COMMAND ml_test_seccomp
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
