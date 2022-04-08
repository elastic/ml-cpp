project("ML Autodetect")

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)
include_directories(SYSTEM ../../3rd_party/eigen)

set(SRCS
Main.cc
CCmdLineParser.cc
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

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
