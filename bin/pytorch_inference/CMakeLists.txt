project("ML PyTorch Inference")

include_directories(../../include)
include_directories(SYSTEM ../../3rd_party/include)
include_directories(SYSTEM ../../3rd_party/rapidjson/include)

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
