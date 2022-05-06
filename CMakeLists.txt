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
cmake_minimum_required(VERSION 3.7)

project("ML")


set (CMAKE_CXX_STANDARD 17)

if(NOT ENV{ML_DEBUG})
  set(OPTCFLAGS "-O3")
  set(OPTCPPFLAGS "-DNDEBUG -DEXCLUDE_TRACE_LOGGING")
endif()

if($ENV{ML_DEBUG})
  if($ENV{ML_COVERAGE})
      set(COVERAGE "--coverage")
  endif()
endif()

set(SNAPSHOT yes)

execute_process(COMMAND uname -m OUTPUT_VARIABLE HARDWARE_ARCH)

execute_process(COMMAND awk -F= "/elasticsearchVersion/ {gsub(/-.*/,\"\"); print $2}" $ENV{CPP_SRC_HOME}/gradle.properties OUTPUT_VARIABLE ML_VERSION_NUM)
string(STRIP "${ML_VERSION_NUM}" ML_VERSION_NUM)
message(STATUS "ML_VERSION_NUM ${ML_VERSION_NUM}")

add_compile_definitions(BOOST_ALL_DYN_LINK BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
add_compile_definitions(RAPIDJSON_HAS_STDSTRING)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(DYNAMIC_LIB_DIR controller.app/Contents/lib)
  set(IMPORT_LIB_DIR lib)

  set(ML_APP_NAME controller)
  message(STATUS "Darwin detected")
  string(REPLACE arm6 aarch64 HARDWARE_ARCH ${HARDWARE_ARCH})
  string(STRIP "${HARDWARE_ARCH}" HARDWARE_ARCH)
  message(STATUS "HARDWARE_ARCH ${HARDWARE_ARCH}")
    
  if(HARDWARE_ARCH STREQUAL "x86_64")
    set(ARCHCFLAGS -msse4.2)
  endif()

  # darwin flags
  if(${CMAKE_CROSSCOMPILING})
    message(STATUS "CROSS_TARGET_PLATFORM=${CROSS_TARGET_PLATFORM}")
    message(STATUS "CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD}")
    set(Boost_COMPILER -clang-darwin11)
    set(SYSROOT /usr/local/sysroot-${CROSS_TARGET_PLATFORM})
    set(CROSS_FLAGS "--sysroot=${SYSROOT} -B /usr/local/bin -target ${CROSS_TARGET_PLATFORM} -stdlib=libc++")
    set(ARCHCFLAGS -msse4.2)
  else()
    set(Boost_COMPILER -clang-darwin13)
    add_compile_definitions(RAPIDJSON_NEON)
  endif()
  set(CMAKE_C_FLAGS "-g ${CROSS_FLAGS} ${OPT_C_FLAGS} ${ARCHFLAGS} -fstack-protector -Weverything -Werror-switch -Wno-deprecated -Wno-disabled-macro-expansion -Wno-documentation-deprecated-sync -Wno-documentation-unknown-command -Wno-extra-semi-stmt -Wno-float-equal -Wno-missing-prototypes -Wno-padded -Wno-poison-system-directories -Wno-sign-conversion -Wno-unknown-warning-option -Wno-unreachable-code -Wno-used-but-marked-unused ${OPTCFLAGS} ${COVERAGE}" )
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-exit-time-destructors -Wno-global-constructors -Wno-return-std-move-in-c++11 -Wno-unused-member-function -Wno-weak-vtables ${OPTCPPFLAGS}" )

  set(CMAKE_INSTALL_RPATH "@loader_path/../lib")

  set(EXE_DIR MacOS)
  set(CMAKE_MACOSX_RPATH 1)
  add_compile_definitions(MacOSX)
  set(PLATFORM_NAME "MacOSX")
  # TODO create simple  cmake module for pytorch
  find_library(TORCH_LIB torch_cpu HINTS /usr/local/lib)
  find_library(C10_LIB c10 HINTS /usr/local/lib)
  find_path(TORCH_INC caffe2/serialize/read_adapter_interface.h)
  include_directories(SYSTEM ${SYSROOT}/usr/include/c++/v1)
  include_directories(SYSTEM ${SYSROOT}/usr/include)
  include_directories(SYSTEM ${SYSROOT}/usr/local/include)
  include_directories(SYSTEM ${SYSROOT}/usr/local/include/pytorch)

  add_custom_target(
      plist ALL
      DEPENDS $ENV{CPP_SRC_HOME}/mk/make_info_plist.sh 
      DEPENDS $ENV{CPP_SRC_HOME}/gradle.properties
      COMMAND $ENV{CPP_SRC_HOME}/mk/make_info_plist.sh "${ML_APP_NAME}" false > Info.plist
  )
  install(FILES ${CMAKE_BINARY_DIR}/Info.plist DESTINATION ${CMAKE_INSTALL_PREFIX} COMPONENT Distributed)
  install(FILES $ENV{CPP_SRC_HOME}/lib/core/ml-en.dict DESTINATION ${CMAKE_INSTALL_PREFIX}/Resources COMPONENT Distributed)

endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "Linux detected")
  set(DYNAMIC_LIB_DIR lib)
  set(IMPORT_LIB_DIR lib)
  # linux flags

  if(${CMAKE_CROSSCOMPILING})
    message(STATUS "CROSS_TARGET_PLATFORM=${CROSS_TARGET_PLATFORM}")
    message(STATUS "CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD}")
    set(SYSROOT /usr/local/sysroot-${CROSS_TARGET_PLATFORM})
    set(CROSS_FLAGS "--sysroot=${SYSROOT}")
    add_compile_definitions(RAPIDJSON_NEON)
  endif()
  
  set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib:\$ORIGIN/")

  execute_process(COMMAND uname -m OUTPUT_VARIABLE HARDWARE_ARCH)
  string(STRIP "${HARDWARE_ARCH}" HARDWARE_ARCH)
  
  if(HARDWARE_ARCH STREQUAL "aarch64")
    message(STATUS "aarch64 detected")
    set(ARCHCFLAGS "-march=armv8-a+crc+crypto")
    add_compile_definitions(RAPIDJSON_NEON)
  else()
    message(STATUS "x86-64 detected")
    set (ARCHCFLAGS "-msse4.2 -mfpmath=sse")
    if(NOT ${CMAKE_CROSSCOMPILING})
      add_compile_definitions(RAPIDJSON_SSE42)
    endif()
  endif()

  set(CMAKE_C_FLAGS "-g ${CROSS_FLAGS} ${OPTCFLAGS} ${ARCHCFLAGS} -fstack-protector -fno-math-errno -Wall -Wcast-align -Wconversion -Wextra -Winit-self -Wno-psabi -Wparentheses -Wpointer-arith -Wswitch-enum")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-ctor-dtor-privacy -Wno-deprecated-declarations -Wold-style-cast -fvisibility-inlines-hidden")

  set(Boost_COMPILER gcc10)
  set(EXE_DIR bin)
  set(PLATFORM_NAME "Linux")
  add_compile_definitions(Linux)

  set(LIBXML2_HINTS /usr/local/gcc103)

  # TODO create simple  cmake module for pytorch
  find_library(TORCH_LIB torch_cpu HINTS /usr/local/gcc103/lib)
  find_library(C10_LIB c10 HINTS /usr/local/gcc103/lib)
  find_path(TORCH_INC caffe2/serialize/read_adapter_interface.h)
  include_directories(SYSTEM ${SYSROOT}/usr/include)
  include_directories(SYSTEM ${SYSROOT}/usr/local/include)
  include_directories(SYSTEM ${SYSROOT}/usr/local/gcc103/include/libxml2)
  include_directories(SYSTEM ${SYSROOT}/usr/local/gcc103/include/pytorch)

  install(FILES $ENV{CPP_SRC_HOME}/lib/core/ml-en.dict DESTINATION ${CMAKE_INSTALL_PREFIX}/resources COMPONENT Distributed)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(DYNAMIC_LIB_DIR bin)
  set(IMPORT_LIB_DIR lib)
  set(OS Windows)
  add_compile_definitions(Windows)
  if(NOT ENV{ML_DEBUG})
    set(OPTCFLAGS "-O2 -Qfast_transcendentals -Qvec-report:1")
    add_compile_definitions(NDEBUG EXCLUDE_TRACE_LOGGING)
  endif()
  
  if(NOT CRT_OPT)
   set(CRT_OPT -MD)
  endif()
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE WIN32_LEAN_AND_MEAN NTDDI_VERSION=0x06010000 _WIN32_WINNT=0x0601 Windows)
  set(CMAKE_C_FLAGS "-nologo ${OPTCFLAGS} -W4 ${CRT_OPT} -EHsc -Zi -Gw -FS -Zc:inline -diagnostics:caret -utf-8")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -TP ${CFLAGS} -Zc:rvalueCast -Zc:strictStrings -wd4127 -we4150 -wd4201 -wd4231 -wd4251 -wd4355 -wd4512 -wd4702 -bigobj ${CPPFLAGS}")

  #take a stick to the compiler flags to get what we need
  set(CompilerFlags
          CMAKE_CXX_FLAGS
          CMAKE_CXX_FLAGS_DEBUG
          CMAKE_CXX_FLAGS_RELEASE
          CMAKE_C_FLAGS
          CMAKE_C_FLAGS_DEBUG
          CMAKE_C_FLAGS_RELEASE
          )
  foreach(CompilerFlag ${CompilerFlags})
    string(REPLACE "/MDd" "/MD" ${CompilerFlag} "${${CompilerFlag}}")
    string(REPLACE "/Od" "/O2" ${CompilerFlag} "${${CompilerFlag}}")
    string(REPLACE "/RTC1" "" ${CompilerFlag} "${${CompilerFlag}}")
  endforeach()

  set(CMAKE_STATIC_LIBRARY_PREFIX lib)
  set(CMAKE_SHARED_LIBRARY_PREFIX lib)
  set(CMAKE_IMPORT_LIBRARY_PREFIX lib)

  install(FILES $ENV{CPP_SRC_HOME}/lib/core/date_time_zonespec.csv DESTINATION ${CMAKE_INSTALL_PREFIX}/resources COMPONENT Distributed)
  install(FILES $ENV{CPP_SRC_HOME}/lib/core/ml-en.dict DESTINATION ${CMAKE_INSTALL_PREFIX}/resources COMPONENT Distributed)

  set(EXE_DIR bin)
  set(PLATFORM_NAME "Windows")
  find_library(TORCH_LIB torch_cpu)
  find_library(C10_LIB c10)
  find_library(STRPTIME_LIB strptime)
  find_library(ZLIB_LIB zdll)
  include_directories(SYSTEM /usr/local/include/pytorch)
  include_directories(SYSTEM c:/usr/local/include)

  set(Boost_USE_DEBUG_LIBS OFF)
  set(Boost_USE_RELEASE_LIBS ON)
  set(Boost_USE_MULTITHREADED ON)
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_STATIC_RUNTIME OFF)
  set(Boost_USE_DEBUG_RUNTIME OFF)
  set(Boost_COMPILER vc142)
  set(Boost_DEBUG OFF)

  add_definitions( -DBOOST_ALL_NO_LIB )
  add_definitions( -DBOOST_ALL_DYN_LINK )
endif()

add_compile_definitions(EIGEN_MPL2_ONLY EIGEN_MAX_ALIGN_BYTES=32)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

list( APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib )

set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_DEBUG OFF)
set(Boost_USE_DEBUG_LIBS OFF)

find_package(Boost 1.77.0 REQUIRED COMPONENTS iostreams filesystem program_options regex date_time log log_setup thread unit_test_framework)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

if (LIBXML2_HINTS)
    find_package(LibXml2 HINTS ${LIBXML2_HINTS})
else()
    find_package(LibXml2)
endif()
if(LIBXML2_FOUND)
  include_directories(${LIBXML2_INCLUDE_DIR})
endif()

find_package(ZLIB)
if(ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIR})
endif()

add_subdirectory(3rd_party)

add_subdirectory(lib/core)
add_subdirectory(lib/maths/common)
add_subdirectory(lib/maths/time_series)
add_subdirectory(lib/maths/analytics)
add_subdirectory(lib/model)
add_subdirectory(lib/api)
add_subdirectory(lib/api/dump_state)
add_subdirectory(lib/test)
add_subdirectory(lib/ver)
add_subdirectory(lib/seccomp)

add_subdirectory(bin/autodetect)
add_subdirectory(bin/categorize)
add_subdirectory(bin/controller)
add_subdirectory(bin/controller/unittest)
add_subdirectory(bin/data_frame_analyzer)
add_subdirectory(bin/normalize)
add_subdirectory(bin/pytorch_inference)
add_subdirectory(bin/pytorch_inference/unittest)

add_subdirectory(lib/core/unittest)
add_subdirectory(lib/maths/common/unittest)
add_subdirectory(lib/maths/time_series/unittest)
add_subdirectory(lib/maths/analytics/unittest)
add_subdirectory(lib/model/unittest)
add_subdirectory(lib/api/unittest)
add_subdirectory(lib/ver/unittest)
add_subdirectory(lib/seccomp/unittest)

install(TARGETS MlApi MlCore MlMathsAnalytics MlMathsCommon MlMathsTimeSeries MlModel MlSeccomp MlTest MlVer autodetect
    categorize controller data_frame_analyzer normalize pytorch_inference COMPONENT Distributed RUNTIME DESTINATION
    ${EXE_DIR})

add_custom_target(install-distributed
  DEPENDS eigen licenses plist MlApi MlCore MlMathsAnalytics MlMathsCommon MlMathsTimeSeries MlModel MlSeccomp MlTest MlVer autodetect categorize controller data_frame_analyzer normalize pytorch_inference
  COMMAND "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=Distributed -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
)

add_custom_target(run_test_model
    DEPENDS ml_test_model
    COMMAND ml_test_model
    WORKING_DIRECTORY ../lib/model/unittest
)
add_custom_target(run_test_core
    DEPENDS ml_test_core
    COMMAND ml_test_core
    WORKING_DIRECTORY ../lib/core/unittest
)
add_custom_target(run_test_api
    DEPENDS ml_test_api
    COMMAND ml_test_api
    WORKING_DIRECTORY ../lib/api/unittest
)
add_custom_target(run_test_seccomp
    DEPENDS ml_test_seccomp
    COMMAND ml_test_seccomp
    WORKING_DIRECTORY ../lib/seccomp/unittest
)
add_custom_target(run_test_maths_common
    DEPENDS ml_test_maths_common
    COMMAND ml_test_maths_common
    WORKING_DIRECTORY ../lib/maths/common/unittest
)
add_custom_target(run_test_maths_time_series
    DEPENDS ml_test_maths_time_series
    COMMAND ml_test_maths_time_series
    WORKING_DIRECTORY ../lib/maths/time_series/unittest
)
add_custom_target(run_test_maths_analytics
    DEPENDS ml_test_maths_analytics
    COMMAND ml_test_maths_analytics
    WORKING_DIRECTORY ../lib/maths/analytics/unittest
)
add_custom_target(run_test_ver
    DEPENDS ml_test_ver
    COMMAND ml_test_ver
    WORKING_DIRECTORY ../lib/ver/unittest
)
add_custom_target(run_test_controller
    DEPENDS ml_test_controller
    COMMAND ml_test_controller
    WORKING_DIRECTORY ../bin/controller/unittest
)
add_custom_target(run_test_pytorch_inference
    DEPENDS ml_test_pytorch_inference
    COMMAND ml_test_pytorch_inference
    WORKING_DIRECTORY ../bin/pytorch_inference/unittest
)
add_custom_target(test
    DEPENDS run_test_model run_test_core run_test_api run_test_seccomp run_test_ver run_test_maths_common
    run_test_maths_time_series run_test_analytics run_test_controller run_test_pytorch_inference
)
