cmake_minimum_required(VERSION 3.7)

project("ML Seccomp")

set(Boost_COMPILER -clang-darwin13)
set (CMAKE_CXX_STANDARD 17)

find_package(Boost 1.77.0 REQUIRED COMPONENTS iostreams filesystem program_options regex date_time log_setup thread unit_test_framework)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()


add_compile_definitions(BOOST_ALL_DYN_LINK BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)

include_directories(../../include)

set(PLATFORM_SRCS
    CSystemCallFilter_MacOSX.cc
)

set(SRCS
)

add_library(MlSeccomp STATIC ${SRCS} ${PLATFORM_SRCS})

#install(TARGETS MlSeccomp DESTINATION lib)
