project("ML DataFrame Analyzer")

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/eigen)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)

set(SRCS
Main.cc
CCmdLineParser.cc
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(data_frame_analyzer ${SRCS})

target_link_libraries(data_frame_analyzer ${Boost_LIBRARIES})
target_link_libraries(data_frame_analyzer MlCore)
target_link_libraries(data_frame_analyzer MlMathsCommon)
target_link_libraries(data_frame_analyzer MlMathsAnalytics)
target_link_libraries(data_frame_analyzer MlModel)
target_link_libraries(data_frame_analyzer MlApi)
target_link_libraries(data_frame_analyzer MlSeccomp)
target_link_libraries(data_frame_analyzer MlVer)

install(TARGETS data_frame_analyzer DESTINATION "${EXE_DIR}")
