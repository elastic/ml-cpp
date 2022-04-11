cmake_minimum_required(VERSION 3.7)

project("ML")

set (CMAKE_CXX_STANDARD 17)

if(NOT $ENV{ML_DEBUG})
  set(OPTCFLAGS -O3)
set(OPTCPPFLAGS -DNDEBUG -DEXCLUDE_TRACE_LOGGING)
endif()

if($ENV{ML_DEBUG})
  if($ENV{ML_COVERAGE})
      set(COVERAGE "--coverage")
  endif()
endif()

execute_process(COMMAND uname -m OUTPUT_VARIABLE HARDWARE_ARCH)

execute_process(COMMAND awk -F= "/elasticsearchVersion/ {gsub(/-.*/,\"\"); print $2}" $ENV{CPP_SRC_HOME}/gradle.properties OUTPUT_VARIABLE ML_VERSION_NUM)
string(STRIP "${ML_VERSION_NUM}" ML_VERSION_NUM)
message(STATUS "ML_VERSION_NUM ${ML_VERSION_NUM}")

add_compile_definitions(BOOST_ALL_DYN_LINK BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
add_compile_definitions(RAPIDJSON_HAS_STDSTRING)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(ML_APP_NAME controller)
  message(STATUS "Darwin detected")
  string(REPLACE arm6 aarch64 HARDWARE_ARCH ${HARDWARE_ARCH})
  string(STRIP "${HARDWARE_ARCH}" HARDWARE_ARCH)
  message(STATUS "HARDWARE_ARCH ${HARDWARE_ARCH}")
    
  if(HARDWARE_ARCH STREQUAL "x86_64")
    set(ARCHCFLAGS -msse4.2)
  endif()

  # darwin flags
  set(CMAKE_C_FLAGS "-g ${OPT_C_FLAGS} ${ARCHFLAGS} -fstack-protector -Weverything -Werror-switch -Wno-deprecated -Wno-disabled-macro-expansion -Wno-documentation-deprecated-sync -Wno-documentation-unknown-command -Wno-float-equal -Wno-missing-prototypes -Wno-padded -Wno-poison-system-directories -Wno-sign-conversion -Wno-unknown-warning-option -Wno-unreachable-code -Wno-used-but-marked-unused" )
  set(CMAKE_C_FLAGS "-g ${OPT_C_FLAGS} ${ARCHFLAGS} -fstack-protector -Weverything -Werror-switch -Wno-deprecated -Wno-disabled-macro-expansion -Wno-documentation-deprecated-sync -Wno-documentation-unknown-command -Wno-float-equal -Wno-missing-prototypes -Wno-padded -Wno-poison-system-directories -Wno-sign-conversion -Wno-unknown-warning-option -Wno-unreachable-code -Wno-used-but-marked-unused ${COVERAGE}" )
  set(Boost_COMPILER -clang-darwin13)
  set(EXE_DIR MacOS)
  set(CMAKE_MACOSX_RPATH 1)
  add_compile_definitions(MacOSX)
  set(PLATFORM_NAME "MacOSX")
  #TODO test for arm
  add_compile_definitions(DRAPIDJSON_NEON)
  # TODO create simple  cmake module for pytorch
  find_library(TORCH_LIB torch_cpu HINTS /usr/local/lib)
  find_library(C10_LIB c10 HINTS /usr/local/lib)
  #find_path(TORCH_INC caffe2/serialize/read_adapter_interface.h)
  include_directories(SYSTEM /usr/local/include/pytorch)

  add_custom_target(
      plist ALL
      DEPENDS $ENV{CPP_SRC_HOME}/mk/make_info_plist.sh 
      DEPENDS $ENV{CPP_SRC_HOME}/gradle.properties
      COMMAND $ENV{CPP_SRC_HOME}/mk/make_info_plist.sh "${ML_APP_NAME}" false > Info.plist
  )
  install(FILES ${CMAKE_BINARY_DIR}/INFO.plist DESTINATION ${CMAKE_INSTALL_PREFIX})
  install(FILES $ENV{CPP_SRC_HOME}/lib/core/ml-en.dict DESTINATION ${CMAKE_INSTALL_PREFIX}/Resources)

endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Linux detected")
  # linux flags
  set(CMAKE_C_FLAGS "-g ${OPTCFLAGS} ${ARCHCFLAGS} -fstack-protector -fno-math-errno -Wall -Wcast-align -Wconversion -Wextra -Winit-self -Wno-psabi -Wparentheses -Wpointer-arith -Wswitch-enum")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-ctor-dtor-privacy -Wno-deprecated-declarations -Wold-style-cast -fvisibility-inlines-hidden")
  
  execute_process(COMMAND uname -m OUTPUT_VARIABLE HARDWARE_ARCH)
  
  if(HARDWARE_ARCH STREQUAL "aarch64")
    message(STATUS "aarch64 detected")
    set(ARCHCFLAGS "-march=armv8-a+crc+crypto")
    add_compile_definitions(DRAPIDJSON_NEON)
  else()
    message(STATUS "x86-64 detected")
    set (ARCHCFLAGS "-msse4.2 -mfpmath=sse")
    add_compile_definitions(DRAPIDJSON_SSE42)
  endif()
  
  set(Boost_COMPILER gcc10)
  set(EXE_DIR bin)
  set(PLATFORM_NAME "Linux")
    # TODO create simple  cmake module for pytorch
  find_library(TORCH_LIB torch_cpu HINTS /usr/local/gcc103/lib)
  find_library(C10_LIB c10 HINTS /usr/local/gcc103/lib)
  #find_path(TORCH_INC caffe2/serialize/read_adapter_interface.h)
  include_directories(SYSTEM /usr/local/gcc103/include/pytorch)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  install(FILES $ENV{CPP_SRC_HOME}/lib/core/date_time_zonespec.csv DESTINATION ${CMAKE_INSTALL_PREFIX}/Resources)
endif()

add_compile_definitions(DEIGEN_MPL2_ONLY DEIGEN_MAX_ALIGN_BYTES=32)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

list( APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib )

set(Boost_USE_MULTITHREADED ON)

set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_DEBUG OFF)

find_package(Boost 1.77.0 REQUIRED COMPONENTS iostreams filesystem program_options regex date_time log log_setup thread unit_test_framework HINTS /usr/local/gcc103 /usr/local)
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

add_subdirectory(3rd_party)

add_subdirectory(lib/core)
add_subdirectory(lib/maths/common)
add_subdirectory(lib/maths/time_series)
add_subdirectory(lib/maths/analytics)
add_subdirectory(lib/model)
add_subdirectory(lib/api)
add_subdirectory(lib/test)
add_subdirectory(lib/ver)
add_subdirectory(lib/seccomp)

add_subdirectory(bin/autodetect)
add_subdirectory(bin/categorize)
add_subdirectory(bin/controller)
add_subdirectory(bin/data_frame_analyzer)
add_subdirectory(bin/normalize)
add_subdirectory(bin/pytorch_inference)

add_subdirectory(lib/core/unittest)
add_subdirectory(lib/maths/common/unittest)
add_subdirectory(lib/maths/time_series/unittest)
add_subdirectory(lib/maths/analytics/unittest)
add_subdirectory(lib/model/unittest)
add_subdirectory(lib/api/unittest)
add_subdirectory(lib/ver/unittest)
add_subdirectory(lib/seccomp/unittest)

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
add_custom_target(test
    DEPENDS run_test_model run_test_core run_test_api run_test_seccomp run_test_ver run_test_maths_common run_test_maths_time_series run_test_analytics
)
