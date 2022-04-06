cmake_minimum_required(VERSION 3.13)

project("ML")

#CFLAGS=-g $(OPTCFLAGS) $(ARCHCFLAGS) -fstack-protector -Weverything -Werror-switch -Wno-deprecated -Wno-disabled-macro-expansion -Wno-documentation-deprecated-sync -Wno-documentation-unknown-command -Wno-float-equal -Wno-missing-prototypes -Wno-padded -Wno-poison-system-directories -Wno-sign-conversion -Wno-unknown-warning-option -Wno-unreachable-code -Wno-used-but-marked-unused $(COVERAGE)
#CXXFLAGS=$(CFLAGS) -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-exit-time-destructors -Wno-global-constructors -Wno-return-std-move-in-c++11 -Wno-unused-member-function -Wno-weak-vtables
#CPPFLAGS=-isystem $(CPP_SRC_HOME)/3rd_party/include -isystem /usr/local/include -D$(OS) $(OPTCPPFLAGS)

if($ENV{ML_DEBUG})
set(OPTCFLAGS -O3)
set(OPTCPPFLAGS -DNDEBUG -DEXCLUDE_TRACE_LOGGING)
endif()

# darwin flags
set(CMAKE_C_FLAGS "-g ${OPT_C_FLAGS} ${ARCHFLAGS} -fstack-protector -Weverything -Werror-switch -Wno-deprecated -Wno-disabled-macro-expansion -Wno-documentation-deprecated-sync -Wno-documentation-unknown-command -Wno-float-equal -Wno-missing-prototypes -Wno-padded -Wno-poison-system-directories -Wno-sign-conversion -Wno-unknown-warning-option -Wno-unreachable-code -Wno-used-but-marked-unused" )
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-exit-time-destructors -Wno-global-constructors -Wno-return-std-move-in-c++11 -Wno-unused-member-function -Wno-weak-vtables" )
#

add_compile_definitions(RAPIDJSON_HAS_STDSTRING DRAPIDJSON_NEON MacOSX)
add_compile_definitions(DEIGEN_MPL2_ONLY DEIGEN_MAX_ALIGN_BYTES=32)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set (CMAKE_CXX_STANDARD 17)

set(CMAKE_MACOSX_RPATH 1)
list( APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib )

set(EXE_DIR MacOS)

set(Boost_USE_MULTITHREADED ON)

set(Boost_COMPILER -clang-darwin13)

set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_DEBUG ON)

find_package(Boost 1.77.0 REQUIRED COMPONENTS iostreams filesystem program_options regex date_time log log_setup thread unit_test_framework)
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
