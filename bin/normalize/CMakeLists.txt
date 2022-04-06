
project("ML Normalize")

include_directories(../../include)
include_directories(../../3rd_party/include)
include_directories(../../3rd_party/eigen)
include_directories(../../3rd_party/rapidjson/include)

add_compile_definitions(RAPIDJSON_HAS_STDSTRING DRAPIDJSON_NEON MacOSX)

set(SRCS
Main.cc
CCmdLineParser.cc
)

add_executable(normalize ${SRCS})

target_link_libraries(normalize ${Boost_LIBRARIES})
target_link_libraries(normalize MlCore)
target_link_libraries(normalize MlModel)
target_link_libraries(normalize MlApi)
target_link_libraries(normalize MlVer)
target_link_libraries(normalize MlSeccomp)

install(TARGETS normalize DESTINATION "${EXE_DIR}")
