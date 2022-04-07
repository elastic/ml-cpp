cmake_minimum_required(VERSION 3.7)

project("ML Seccomp")

set (CMAKE_CXX_STANDARD 17)

find_package(Boost 1.77.0 REQUIRED COMPONENTS iostreams filesystem program_options regex date_time log log_setup thread unit_test_framework)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

add_compile_definitions(BOOST_ALL_DYN_LINK BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
add_compile_definitions(RAPIDJSON_HAS_STDSTRING DRAPIDJSON_NEON)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
add_compile_definitions(MacOSX)
set(Boost_COMPILER -clang-darwin13)
set(CMAKE_MACOSX_RPATH 1)
set(PLATFORM_NAME "MacOSX")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
set(Boost_COMPILER gcc10)
set(PLATFORM_NAME "Linux")
endif()

list( APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib )


include_directories(../../include)

set(PLATFORM_SRCS
    CSystemCallFilter_${PLATFORM_NAME}.cc
)

set(SRCS
)

add_library(MlSeccomp STATIC ${SRCS} ${PLATFORM_SRCS})
