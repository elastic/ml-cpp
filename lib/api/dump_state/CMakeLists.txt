project("ML Autodetect")

include_directories(../../../include)
include_directories(SYSTEM ../../../3rd_party/include)
include_directories(SYSTEM ../../../3rd_party/rapidjson/include)
include_directories(SYSTEM ../../../3rd_party/eigen)

set(SRCS
Main.cc
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(dump_state ${SRCS})

target_link_libraries(dump_state ${Boost_LIBRARIES})
target_link_libraries(dump_state MlCore)
target_link_libraries(dump_state MlApi)
target_link_libraries(dump_state MlSeccomp)
target_link_libraries(dump_state MlVer)

#install(TARGETS dump_state DESTINATION "${EXE_DIR}")
