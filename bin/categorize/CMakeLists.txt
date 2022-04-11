project("ML Categorize")

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)
include_directories(SYSTEM ../../3rd_party/eigen)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)

set(SRCS
Main.cc
CCmdLineParser.cc
)

add_executable(categorize ${SRCS})

target_link_libraries(categorize ${Boost_LIBRARIES})
target_link_libraries(categorize MlCore)
target_link_libraries(categorize MlMathsCommon)
target_link_libraries(categorize MlModel)
target_link_libraries(categorize MlApi)
target_link_libraries(categorize MlSeccomp)
target_link_libraries(categorize MlVer)

install(TARGETS categorize DESTINATION "${EXE_DIR}")
