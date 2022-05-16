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
cmake_minimum_required(VERSION 3.9)

# Set these here to avoid any confusion. See the documentation
# in build-setup for more details
if (WIN32)
  set(CMAKE_MAKE_PROGRAM "gnumake.exe" CACHE STRING "" FORCE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  set(CMAKE_C_COMPILER "cl.exe" CACHE STRING "" FORCE)
  set(CMAKE_CXX_COMPILER "cl.exe" CACHE STRING "" FORCE)
elseif(UNIX AND NOT APPLE)
  set(CMAKE_C_COMPILER   "gcc" CACHE STRING "" FORCE)
  set(CMAKE_CXX_COMPILER "g++" CACHE STRING "" FORCE)
endif()

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

set(CMAKE_INSTALL_PREFIX "$ENV{CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}")

execute_process(COMMAND awk -F= "/elasticsearchVersion/ {gsub(/-.*/,\"\"); print $2}" $ENV{CPP_SRC_HOME}/gradle.properties OUTPUT_VARIABLE ML_VERSION_NUM)
string(STRIP "${ML_VERSION_NUM}" ML_VERSION_NUM)
message(STATUS "ML_VERSION_NUM ${ML_VERSION_NUM}")

message(STATUS "CMAKE_CXX_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION}")
STRING(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" CMAKE_CXX_COMPILER_VERSION_REGEX_MATCH  ${CMAKE_CXX_COMPILER_VERSION})
set(CMAKE_CXX_COMPILER_VERSION_MAJOR ${CMAKE_MATCH_1})
set(CMAKE_CXX_COMPILER_VERSION_MINOR ${CMAKE_MATCH_2})
set(CMAKE_CXX_COMPILER_VERSION_PATCH ${CMAKE_MATCH_3})

add_compile_definitions(BOOST_ALL_DYN_LINK BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
add_compile_definitions(RAPIDJSON_HAS_STDSTRING)

function(gen_platform_srcs SRCS_LIST)
    foreach(FILE ${SRCS_LIST})
      set(FILE "${CMAKE_CURRENT_SOURCE_DIR}/${FILE}")
      string(REPLACE ".cc" "_${PLATFORM_NAME}.cc" FILE_TMP ${FILE})
      if (EXISTS ${FILE_TMP})
          list(APPEND PLATFORM_SRCS ${FILE_TMP})
      else()
          list(APPEND PLATFORM_SRCS ${FILE})
      endif()
  endforeach()
  set(PLATFORM_SRCS ${PLATFORM_SRCS} PARENT_SCOPE)
endfunction()

set(INSTALL_DEPENDS eigen licenses MlApi MlCore MlMathsAnalytics MlMathsCommon MlMathsTimeSeries MlModel MlSeccomp
    MlTest MlVer autodetect categorize controller data_frame_analyzer normalize pytorch_inference)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(DYNAMIC_LIB_DIR controller.app/Contents/lib)
  set(IMPORT_LIB_DIR lib)

  set(ML_APP_NAME controller)
  message(STATUS "Darwin detected")
    
  message(STATUS "CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR}")
  if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^aarch" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "^arm")
    message(STATUS "aarch64 detected for target")
    set(ARCHCFLAGS "-march=armv8-a+crc+crypto")
    add_compile_definitions(RAPIDJSON_NEON)
  else()
    message(STATUS "x86_64 detected for target")
    set (ARCHCFLAGS "-msse4.2")
    add_compile_definitions(RAPIDJSON_SSE42)
  endif()
  

  # darwin flags
  if(${CMAKE_CROSSCOMPILING})
    message(STATUS "CROSS_TARGET_PLATFORM=${CROSS_TARGET_PLATFORM}")
    message(STATUS "CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD}")
    set(SYSROOT /usr/local/sysroot-${CROSS_TARGET_PLATFORM})
    set(CROSS_FLAGS "--sysroot=${SYSROOT} -B /usr/local/bin -target ${CROSS_TARGET_PLATFORM} -stdlib=libc++")

    include_directories(SYSTEM ${SYSROOT}/usr/include/c++/v1)
    include_directories(SYSTEM ${SYSROOT}/usr/include)
    include_directories(SYSTEM ${SYSROOT}/usr/include/libxml2)

    set(Boost_COMPILER "-clang-darwin11")
  else()
    set(Boost_COMPILER "-clang-darwin${CMAKE_CXX_COMPILER_VERSION_MAJOR}")
  endif()
  message(STATUS "Boost_COMPILER ${Boost_COMPILER}")


  set(CMAKE_C_FLAGS "-g ${CROSS_FLAGS} ${OPT_C_FLAGS} ${ARCHCFLAGS} -fstack-protector -Weverything -Werror-switch -Wno-deprecated -Wno-disabled-macro-expansion -Wno-documentation-deprecated-sync -Wno-documentation-unknown-command -Wno-extra-semi-stmt -Wno-float-equal -Wno-missing-prototypes -Wno-padded -Wno-poison-system-directories -Wno-sign-conversion -Wno-unknown-warning-option -Wno-unreachable-code -Wno-used-but-marked-unused ${OPTCFLAGS} ${COVERAGE}" )
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-exit-time-destructors -Wno-global-constructors -Wno-return-std-move-in-c++11 -Wno-unused-member-function -Wno-weak-vtables ${OPTCPPFLAGS}" )

  set(EXE_DIR MacOS)
  set(CMAKE_MACOSX_RPATH 1)
  set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
  add_compile_definitions(MacOSX)
  set(PLATFORM_NAME "MacOSX")

  # Hardcode paths to avoid any confusion
  set(TORCH_INC "${SYSROOT}/usr/local/include/pytorch")
  set(TORCH_LIB "${SYSROOT}/usr/local/lib/libtorch_cpu.dylib")
  set(C10_LIB   "${SYSROOT}/usr/local/lib/libc10.dylib")
  set(ZLIB_LIBRARIES "-lz")
  set(LIBXML2_LIBRARIES "-lxml2")

  include_directories(SYSTEM ${SYSROOT}/usr/local/include)

  add_custom_target(
      plist ALL
      DEPENDS $ENV{CPP_SRC_HOME}/mk/make_info_plist.sh 
      DEPENDS $ENV{CPP_SRC_HOME}/gradle.properties
      COMMAND $ENV{CPP_SRC_HOME}/mk/make_info_plist.sh "${ML_APP_NAME}" false > Info.plist
  )
  install(FILES ${CMAKE_BINARY_DIR}/Info.plist DESTINATION ${CMAKE_INSTALL_PREFIX} COMPONENT Distributed)
  install(FILES $ENV{CPP_SRC_HOME}/lib/core/ml-en.dict DESTINATION ${CMAKE_INSTALL_PREFIX}/Resources COMPONENT Distributed)

  list( APPEND INSTALL_DEPENDS plist )
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
  endif()

  # Target architecture specific flags
  message(STATUS "CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR}")
  if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^aarch" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "^arm")
    message(STATUS "aarch64 detected for target")
    set(ARCHCFLAGS "-march=armv8-a+crc+crypto")
    add_compile_definitions(RAPIDJSON_NEON)
  else()
    message(STATUS "x86_64 detected for target")
    set (ARCHCFLAGS "-msse4.2 -mfpmath=sse")
    add_compile_definitions(RAPIDJSON_SSE42)
  endif()
  
  set(CMAKE_C_FLAGS "-g ${CROSS_FLAGS} ${OPTCFLAGS} ${ARCHCFLAGS} -fstack-protector -fno-math-errno -Wall -Wcast-align -Wconversion -Wextra -Winit-self -Wno-psabi -Wparentheses -Wpointer-arith -Wswitch-enum")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-ctor-dtor-privacy -Wno-deprecated-declarations -Wold-style-cast -fvisibility-inlines-hidden")

  set(Boost_COMPILER "gcc${CMAKE_CXX_COMPILER_VERSION_MAJOR}")
  set(EXE_DIR bin)
  set(PLATFORM_NAME "Linux")
  add_compile_definitions(Linux)

  set(LIBXML2_HINTS /usr/local/gcc103)

  # Hardcode paths to avoid any confusion
  set(TORCH_INC "/usr/local/gcc103/include/pytorch")
  set(TORCH_LIB "/usr/local/gcc103/lib/libtorch_cpu.so")
  set(C10_LIB   "/usr/local/gcc103/lib/libc10.so")
  set(ZLIB_LIBRARIES "-lz")
  set(LIBXML2_LIBRARIES "-lxml2")

  include_directories(SYSTEM ${SYSROOT}/usr/include)
  include_directories(SYSTEM ${SYSROOT}/usr/local/include)
  include_directories(SYSTEM ${SYSROOT}/usr/local/gcc103/include/libxml2)

  install(FILES $ENV{CPP_SRC_HOME}/lib/core/ml-en.dict DESTINATION ${CMAKE_INSTALL_PREFIX}/resources COMPONENT Distributed)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")

  message(STATUS "VCVER $ENV{VCVER}")
  STRING(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" VCVER_REGEX_MATCH  $ENV{VCVER})
  set(VCVER_MAJOR ${CMAKE_MATCH_1})
  set(VCVER_MINOR ${CMAKE_MATCH_2})

  string(CONCAT BOOST_VCVER "vc" ${VCVER_MAJOR} ${VCVER_MINOR} )
  string(SUBSTRING ${BOOST_VCVER} 0 5 BOOST_VCVER)
  message(STATUS "BOOST_VCVER ${BOOST_VCVER}")

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

  set(CMAKE_STATIC_LIBRARY_PREFIX lib)
  set(CMAKE_SHARED_LIBRARY_PREFIX lib)
  set(CMAKE_IMPORT_LIBRARY_PREFIX lib)

  install(FILES $ENV{CPP_SRC_HOME}/lib/core/date_time_zonespec.csv DESTINATION ${CMAKE_INSTALL_PREFIX}/resources COMPONENT Distributed)
  install(FILES $ENV{CPP_SRC_HOME}/lib/core/ml-en.dict DESTINATION ${CMAKE_INSTALL_PREFIX}/resources COMPONENT Distributed)

  set(EXE_DIR bin)
  set(PLATFORM_NAME "Windows")

  # Hardcode paths to avoid any confusion
  set(LOCAL_DRIVE "$ENV{LOCAL_DRIVE}:")
  message(STATUS "LOCAL_DRIVE ${LOCAL_DRIVE}")
  set(TORCH_INC "${LOCAL_DRIVE}/usr/local/include/pytorch")
  set(TORCH_LIB "${LOCAL_DRIVE}/usr/local/lib/torch_cpu.lib")
  set(C10_LIB "${LOCAL_DRIVE}/usr/local/lib/c10.lib")
  set(STRPTIME_LIB "${LOCAL_DRIVE}/usr/local/lib/strptime.lib")
  set(ZLIB_LIBRARIES "${LOCAL_DRIVE}/usr/local/lib/zlib.lib")
  set(LIBXML2_LIBRARIES "${LOCAL_DRIVE}/usr/local/lib/libxml2.lib")

  include_directories(SYSTEM "${LOCAL_DRIVE}/usr/local/include")
  include_directories(SYSTEM "${LOCAL_DRIVE}/usr/local/include/libxml2")

  set(Boost_USE_DEBUG_LIBS OFF)
  set(Boost_USE_RELEASE_LIBS ON)
  set(Boost_USE_MULTITHREADED ON)
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_STATIC_RUNTIME OFF)
  set(Boost_USE_DEBUG_RUNTIME OFF)
  set(Boost_COMPILER ${BOOST_VCVER})
  set(Boost_DEBUG OFF)

  add_compile_definitions(BOOST_ALL_NO_LIB BOOST_ALL_DYN_LINK)
endif()

include_directories(SYSTEM ${TORCH_INC})

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
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
endif()

add_subdirectory(3rd_party)

add_subdirectory(lib)
add_subdirectory(bin)

install(TARGETS MlApi MlCore MlMathsAnalytics MlMathsCommon MlMathsTimeSeries MlModel MlSeccomp MlTest MlVer autodetect
    categorize controller data_frame_analyzer normalize pytorch_inference COMPONENT Distributed RUNTIME DESTINATION
    ${EXE_DIR})

add_custom_target(install-distributed
    DEPENDS ${INSTALL_DEPENDS}
  COMMAND "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=Distributed -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
)

add_custom_target(test
    DEPENDS run_test_model run_test_core run_test_api run_test_seccomp run_test_ver run_test_maths_common
    run_test_maths_time_series run_test_maths_analytics run_test_controller run_test_pytorch_inference
)
