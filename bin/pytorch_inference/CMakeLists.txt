cmake_minimum_required(VERSION 3.7)
project("ML PyTorch Inference")

cmake_minimum_required(VERSION 3.7)
set (CMAKE_CXX_STANDARD 17)

include_directories(../../include)
include_directories(../../3rd_party/include)
include_directories(../../3rd_party/rapidjson/include)

add_compile_definitions(RAPIDJSON_HAS_STDSTRING)
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Darwin detected")
  # darwin flags
  set(CMAKE_C_FLAGS "-g ${OPT_C_FLAGS} ${ARCHFLAGS} -fstack-protector -Weverything -Werror-switch -Wno-deprecated -Wno-disabled-macro-expansion -Wno-documentation-deprecated-sync -Wno-documentation-unknown-command -Wno-float-equal -Wno-missing-prototypes -Wno-padded -Wno-poison-system-directories -Wno-sign-conversion -Wno-unknown-warning-option -Wno-unreachable-code -Wno-used-but-marked-unused" )
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-exit-time-destructors -Wno-global-constructors -Wno-return-std-move-in-c++11 -Wno-unused-member-function -Wno-weak-vtables" )
  set(Boost_COMPILER -clang-darwin13)
  set(EXE_DIR MacOS)
  set(CMAKE_MACOSX_RPATH 1)
#TODO test for arm
  add_compile_definitions(DRAPIDJSON_NEON)
  # TODO create simple  cmake module for pytorch
  find_library(TORCH_LIB torch_cpu HINTS /usr/local/lib)
  find_library(C10_LIB c10 HINTS /usr/local/lib)
  #find_path(TORCH_INC caffe2/serialize/read_adapter_interface.h)
  include_directories(/usr/local/include/pytorch)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Linux detected")
  # linux flags
  set(CMAKE_C_FLAGS "-g ${OPTCFLAGS} ${ARCHCFLAGS} -fstack-protector -fno-math-errno -Wall -Wcast-align -Wconversion -Wextra -Winit-self -Wno-psabi -Wparentheses -Wpointer-arith -Wswitch-enum")
  #set(CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS} "-Wno-ctor-dtor-privacy -Wno-deprecated-declarations -Wold-style-cast -fvisibility-inlines-hidden")
  
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
  # TODO create simple  cmake module for pytorch
  find_library(TORCH_LIB torch_cpu HINTS /usr/local/gcc103/lib)
  find_library(C10_LIB c10 HINTS /usr/local/gcc103/lib)
  #find_path(TORCH_INC caffe2/serialize/read_adapter_interface.h)
  include_directories(/usr/local/gcc103/include/pytorch)
endif()

set(Boost_USE_MULTITHREADED ON)

set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_DEBUG ON)

find_package(Boost 1.77.0 REQUIRED COMPONENTS iostreams filesystem program_options regex date_time log thread unit_test_framework)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()


link_directories(${CMAKE_INSTALL_PREFIX}/lib)

set(SRCS
Main.cc
CBufferedIStreamAdapter.cc
CCmdLineParser.cc
CCommandParser.cc
SettingsValidator.cc
)

add_executable(pytorch_inference ${SRCS})

target_link_libraries(pytorch_inference ${Boost_LIBRARIES})
target_link_libraries(pytorch_inference MlCore)
target_link_libraries(pytorch_inference MlApi)
target_link_libraries(pytorch_inference MlSeccomp)
target_link_libraries(pytorch_inference MlVer)
target_link_libraries(pytorch_inference "${TORCH_LIB}")
target_link_libraries(pytorch_inference "${C10_LIB}")
#set_property(TARGET  pytorch_inference PROPERTY CXX_STANDARD 14)

install(TARGETS pytorch_inference DESTINATION "${EXE_DIR}")
