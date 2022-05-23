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

# set the C++ standard we need to enforce
set (CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use")
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

set(SNAPSHOT yes)

set(ML_APP_NAME controller)

execute_process(COMMAND awk -F= "/elasticsearchVersion/ {gsub(/-.*/,\"\"); print $2}"
    $ENV{CPP_SRC_HOME}/gradle.properties OUTPUT_VARIABLE ML_VERSION_NUM OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "ML_VERSION_NUM ${ML_VERSION_NUM}")

message(STATUS "CMAKE_CXX_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION}")
STRING(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" CMAKE_CXX_COMPILER_VERSION_REGEX_MATCH  ${CMAKE_CXX_COMPILER_VERSION})
set(CMAKE_CXX_COMPILER_VERSION_MAJOR ${CMAKE_MATCH_1})
set(CMAKE_CXX_COMPILER_VERSION_MINOR ${CMAKE_MATCH_2})
set(CMAKE_CXX_COMPILER_VERSION_PATCH ${CMAKE_MATCH_3})

set(ML_COMPILE_DEFINITIONS BOOST_ALL_DYN_LINK BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS RAPIDJSON_HAS_STDSTRING)

if(NOT DEFINED ENV{CMAKE_INSTALL_PREFIX})
  if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_INSTALL_PREFIX $ENV{CPP_PLATFORM_HOME}/controller.app/Contents/)
  else()
    set(CMAKE_INSTALL_PREFIX $ENV{CPP_PLATFORM_HOME})
  endif()
else()
  set(CMAKE_INSTALL_PREFIX $ENV{CMAKE_INSTALL_PREFIX})
endif()

  
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  if (CMAKE_CROSSCOMPILING)
    set(ML_BOOST_COMPILER_VER "-clang-darwin11")
  else()
    set(ML_BOOST_COMPILER_VER "-clang-darwin${CMAKE_CXX_COMPILER_VERSION_MAJOR}")
  endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(ML_BOOST_COMPILER_VER "gcc${CMAKE_CXX_COMPILER_VERSION_MAJOR}")
else()
  string(CONCAT BOOST_VCVER "vc" ${VCVER_MAJOR} ${VCVER_MINOR} )
  string(SUBSTRING ${BOOST_VCVER} 0 5 BOOST_VCVER)
  message(STATUS "BOOST_VCVER ${BOOST_VCVER}")

  set(ML_BOOST_COMPILER_VER ${BOOST_VCVER})
endif()
message(STATUS "ML_BOOST_COMPILER_VER ${ML_BOOST_COMPILER_VER}")
#message(STATUS "Boost_COMPILER ${Boost_COMPILER}")

set(ML_BASE_PATH ${SYSROOT}/usr/local/)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  string(APPEND ML_BASE_PATH "gcc103")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(ML_LIBEXT ".so")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(ML_LIBEXT ".dylib")
else()
  set(ML_LIBEXT ".lib")
endif()

set(TORCH_INC "${ML_BASE_PATH}/include/pytorch")
set(TORCH_LIB "${ML_BASE_PATH}/lib/libtorch_cpu${ML_LIBEXT}")
set(C10_LIB   "${ML_BASE_PATH}/lib/libc10${ML_LIBEXT}")

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(LIBXML2_LIBRARIES "-lxml2")
else()
  set(LIBXML2_LIBRARIES "${ML_BASE_PATH}/lib/libxml2${ML_LIBEXT}")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(ZLIB_LIBRARIES "${ML_BASE_PATH}/lib/zlib${ML_LIBEXT}")
else()
  set(ZLIB_LIBRARIES "-lz")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(STRPTIME_LIB "${ML_BASE_PATH}/lib/strptime${ML_LIBEXT}")
endif()


if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  if (CMAKE_CROSSCOMPILING)
    list(APPEND ML_SYSTEM_INCLUDE_DIRECTORIES ${SYSROOT}/usr/include/c++/v1)
  endif()
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND ML_SYSTEM_INCLUDE_DIRECTORIES ${ML_BASE_PATH}/include/libxml2)
endif()
list(APPEND ML_SYSTEM_INCLUDE_DIRECTORIES ${SYSROOT}/usr/include ${SYSROOT}/usr/include/libxml2
    ${SYSROOT}/usr/local/include ${TORCH_INC})

set(IMPORT_LIB_DIR lib)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(DYNAMIC_LIB_DIR lib)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(DYNAMIC_LIB_DIR controller.app/Contents/lib)
else()
  set(DYNAMIC_LIB_DIR bin)
endif()


if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(ML_RESOURCES_DIR ${CMAKE_INSTALL_PREFIX}/Resources)
endif()


if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  list(APPEND ML_COMPILE_DEFINITIONS BOOST_ALL_NO_LIB)
endif()

list(APPEND ML_COMPILE_DEFINITIONS EIGEN_MPL2_ONLY EIGEN_MAX_ALIGN_BYTES=32)

set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_DEBUG ON)
set(Boost_USE_DEBUG_LIBS OFF)
set(Boost_USE_RELEASE_LIBS ON)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_DEBUG_RUNTIME OFF)
set(Boost_COMPILER ${ML_BOOST_COMPILER_VER})


find_package(Boost 1.77.0 REQUIRED COMPONENTS iostreams filesystem program_options regex date_time log log_setup thread unit_test_framework)
if(Boost_FOUND)
    list(APPEND ML_SYSTEM_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
endif()

set(Boost_LIBRARIES_WITH_UNIT_TEST ${Boost_LIBRARIES})
list(REMOVE_ITEM Boost_LIBRARIES "Boost::unit_test_framework")

