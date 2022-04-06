project("ML PyTorch Inference")

# TODO create simple  cmake module for pytorch
find_library(TORCH_LIB torch_cpu)
find_library(C10_LIB c10)
#find_path(TORCH_INC caffe2/serialize/read_adapter_interface.h)

include_directories(../../include)
include_directories(../../3rd_party/include)
include_directories(../../3rd_party/rapidjson/include)
include_directories(/usr/local/include/pytorch)

add_compile_definitions(RAPIDJSON_HAS_STDSTRING DRAPIDJSON_NEON MacOSX)

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
