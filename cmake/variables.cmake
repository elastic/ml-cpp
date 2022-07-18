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

file(READ ${CMAKE_SOURCE_DIR}/gradle.properties GRADLE_PROPERTIES)
if(${GRADLE_PROPERTIES} MATCHES "elasticsearchVersion=([0-9\.]+)")
  set(ML_VERSION_NUM "${CMAKE_MATCH_1}")
endif()

message(STATUS "ML_VERSION_NUM ${ML_VERSION_NUM}")

message(STATUS "CMAKE_CXX_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION}")
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" CMAKE_CXX_COMPILER_VERSION_REGEX_MATCH  ${CMAKE_CXX_COMPILER_VERSION})
set(CMAKE_CXX_COMPILER_VERSION_MAJOR ${CMAKE_MATCH_1})
set(CMAKE_CXX_COMPILER_VERSION_MINOR ${CMAKE_MATCH_2})
set(CMAKE_CXX_COMPILER_VERSION_PATCH ${CMAKE_MATCH_3})

list(APPEND ML_COMPILE_DEFINITIONS BOOST_ALL_DYN_LINK BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS RAPIDJSON_HAS_STDSTRING)

if(NOT DEFINED ENV{CMAKE_INSTALL_PREFIX})
  if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_INSTALL_PREFIX "${CPP_PLATFORM_HOME}/controller.app/Contents/")
  else()
    set(CMAKE_INSTALL_PREFIX "${CPP_PLATFORM_HOME}")
  endif()
else()
  set(CMAKE_INSTALL_PREFIX "$ENV{CMAKE_INSTALL_PREFIX}")
endif()

string(REPLACE "\\" "/" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  if (CMAKE_CROSSCOMPILING)
    set(ML_BOOST_COMPILER_VER "-clang-darwin11")
  else()
    # For macOS we usually only use a particular version as our build platform
    # once Xcode has stopped receiving updates for it. However, with Big Sur
    # on ARM we couldn't do this, as Big Sur was the first macOS version for
    # ARM. Therefore, the compiler may get upgraded on a CI server, and we
    # need to hardcode the version that was used to build Boost for that
    # version of Elasticsearch.
    if(DEFINED ENV{BOOSTCLANGVER})
      set(ML_BOOST_COMPILER_VER "-clang-darwin$ENV{BOOSTCLANGVER}")
    else()
      set(ML_BOOST_COMPILER_VER "-clang-darwin${CMAKE_CXX_COMPILER_VERSION_MAJOR}")
    endif()
  endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(ML_BOOST_COMPILER_VER "gcc${CMAKE_CXX_COMPILER_VERSION_MAJOR}")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  string(CONCAT BOOST_VCVER "vc" ${VCVER_MAJOR} ${VCVER_MINOR} )
  string(SUBSTRING ${BOOST_VCVER} 0 5 BOOST_VCVER)
  message(STATUS "BOOST_VCVER ${BOOST_VCVER}")

  set(ML_BOOST_COMPILER_VER ${BOOST_VCVER})
  if(NOT BOOST_ROOT AND NOT DEFINED ENV{BOOST_ROOT})
    set(BOOST_ROOT ${ROOT}/usr/local)
  endif()
else()
  message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()
message(STATUS "ML_BOOST_COMPILER_VER ${ML_BOOST_COMPILER_VER}")

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(ML_BASE_PATH ${ROOT}/usr/local/)
else()
  set(ML_BASE_PATH ${SYSROOT}/usr/local/)
endif()

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
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(TORCH_LIB "${ML_BASE_PATH}/lib/torch_cpu${ML_LIBEXT}")
  set(C10_LIB   "${ML_BASE_PATH}/lib/c10${ML_LIBEXT}")
else()
  set(TORCH_LIB "${ML_BASE_PATH}/lib/libtorch_cpu${ML_LIBEXT}")
  set(C10_LIB   "${ML_BASE_PATH}/lib/libc10${ML_LIBEXT}")
endif()

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
  list(APPEND ML_SYSTEM_INCLUDE_DIRECTORIES ${SYSROOT}/usr/include ${SYSROOT}/usr/include/libxml2 ${SYSROOT}/usr/local/include)
else ()
  list(APPEND ML_SYSTEM_INCLUDE_DIRECTORIES ${ML_BASE_PATH}/include/libxml2)
endif()

list(APPEND ML_SYSTEM_INCLUDE_DIRECTORIES
  ${TORCH_INC}
  ${CMAKE_SOURCE_DIR}/3rd_party/include
  ${CMAKE_SOURCE_DIR}/3rd_party/eigen
  ${CMAKE_SOURCE_DIR}/3rd_party/rapidjson/include
  )

set(IMPORT_LIB_DIR lib)
if (WIN32)
  set(DYNAMIC_LIB_DIR bin)
else()
  set(DYNAMIC_LIB_DIR lib)
endif()


if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(ML_RESOURCES_DIR ${CMAKE_INSTALL_PREFIX}/Resources)
else()
  set(ML_RESOURCES_DIR ${CMAKE_INSTALL_PREFIX}/resources)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  list(APPEND ML_COMPILE_DEFINITIONS BOOST_ALL_NO_LIB)
endif()

# Dictate which flags to use for "Release" and "Debug" builds
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /D NDEBUG /D EXCLUDE_TRACE_LOGGING /Qfast_transcendentals /Qvec-report:1")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/Zi /O2 /D NDEBUG /D EXCLUDE_TRACE_LOGGING /Qfast_transcendentals /Qvec-report:1")
  set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od /RTC1")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DEXCLUDE_TRACE_LOGGING -Wdisabled-optimization -D_FORTIFY_SOURCE=2")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3 -DNDEBUG -DEXCLUDE_TRACE_LOGGING -Wdisabled-optimization -D_FORTIFY_SOURCE=2")
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DEXCLUDE_TRACE_LOGGING")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3 -DNDEBUG -DEXCLUDE_TRACE_LOGGING")
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
endif()
message(STATUS "CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")

# Perform a "RelWithDebInfo" build by default...
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# However, to keep in step with our historical
# build system if ML_DEBUG is set to a boolean
# true value, e.g. ML_DEBUG=1, then override
# the build type to be Debug for single-config
# CMake generators (such as Unix Makefiles).
# For multi-config generators (such as Visual Studio)
# the CMAKE_BUILD_TYPE variable is ignored and instead the
# build type is determined at build time via the '--config' option.
# See https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html
# and https://cmake.org/cmake/help/latest/variable/CMAKE_CONFIGURATION_TYPES.html
# for more detail.
if("$ENV{ML_DEBUG}")
  set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

if(UNIX AND CMAKE_BUILD_TYPE STREQUAL Debug AND DEFINED ENV{ML_COVERAGE})
  set(ML_COVERAGE "--coverage")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND ML_COMPILE_DEFINITIONS "_REENTRANT")
endif()

list(APPEND ML_COMPILE_DEFINITIONS EIGEN_MPL2_ONLY EIGEN_MAX_ALIGN_BYTES=32)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  list(APPEND ML_COMPILE_DEFINITIONS EIGEN_VECTORIZE_SSE3 EIGEN_VECTORIZE_SSE4_1 EIGEN_VECTORIZE_SSE4_2)
endif()

set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_DEBUG OFF)
set(Boost_USE_DEBUG_LIBS OFF)
set(Boost_USE_RELEASE_LIBS ON)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_DEBUG_RUNTIME OFF)
set(Boost_COMPILER "${ML_BOOST_COMPILER_VER}")

find_package(Boost 1.77.0 REQUIRED COMPONENTS iostreams filesystem program_options regex date_time log log_setup thread unit_test_framework)
if(Boost_FOUND)
  list(APPEND ML_SYSTEM_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
endif()

set(Boost_LIBRARIES_WITH_UNIT_TEST ${Boost_LIBRARIES})
list(REMOVE_ITEM Boost_LIBRARIES "Boost::unit_test_framework")

if(MSVC)
  set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
  set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
  set(CMAKE_IMPORT_LIBRARY_PREFIX "lib")
endif()

string(TIMESTAMP BUILD_YEAR "%Y")

if(WIN32)
  set(ML_USER $ENV{USERNAME})
else()
  execute_process(COMMAND id COMMAND awk -F ")" "{ print $1 }" COMMAND awk -F "(" "{ print $2 }" OUTPUT_VARIABLE ML_USER
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

execute_process(COMMAND git rev-parse --short=14 HEAD WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}  OUTPUT_VARIABLE ML_BUILD_STR OUTPUT_STRIP_TRAILING_WHITESPACE)
