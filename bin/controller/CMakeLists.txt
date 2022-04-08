project("ML Controller")

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)
include_directories(SYSTEM ../../3rd_party/eigen)

set(SRCS
Main.cc
CBlockingCallCancellingStreamMonitor.cc
CCmdLineParser.cc
CCommandProcessor.cc
CResponseJsonWriter.cc
)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(controller ${SRCS})

target_link_libraries(controller ${Boost_LIBRARIES})
target_link_libraries(controller MlCore)
target_link_libraries(controller MlVer)
target_link_libraries(controller MlSeccomp)

install(TARGETS controller DESTINATION "${EXE_DIR}")
