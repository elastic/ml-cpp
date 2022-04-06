project("ML Autodetect")

cmake_minimum_required(VERSION 3.13)
set (CMAKE_CXX_STANDARD 17)

set(Boost_COMPILER -clang-darwin13)
find_package(Boost 1.77.0 REQUIRED COMPONENTS iostreams filesystem program_options regex date_time log thread unit_test_framework)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()


include_directories(../../include)
include_directories(../../3rd_party/include)
include_directories(../../3rd_party/rapidjson/include)
include_directories(../../3rd_party/eigen)

add_compile_definitions(RAPIDJSON_HAS_STDSTRING DRAPIDJSON_NEON MacOSX)

set(SRCS
Main.cc
CCmdLineParser.cc
)

add_executable(autodetect ${SRCS})

target_link_libraries(autodetect ${Boost_LIBRARIES})
target_link_libraries(autodetect MlCore)
target_link_libraries(autodetect MlMathsCommon)
target_link_libraries(autodetect MlMathsTimeSeries)
target_link_libraries(autodetect MlModel)
target_link_libraries(autodetect MlApi)
target_link_libraries(autodetect MlSeccomp)
target_link_libraries(autodetect MlVer)

install(TARGETS autodetect DESTINATION "${EXE_DIR}")
