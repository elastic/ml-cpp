if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
endif()

add_subdirectory(autodetect)
add_subdirectory(categorize)
add_subdirectory(controller)
add_subdirectory(controller/unittest)
add_subdirectory(data_frame_analyzer)
add_subdirectory(normalize)
add_subdirectory(pytorch_inference)
add_subdirectory(pytorch_inference/unittest)

add_custom_target(run_test_controller
    DEPENDS ml_test_controller
    COMMAND ml_test_controller
    WORKING_DIRECTORY ../../bin/controller/unittest
)
add_custom_target(run_test_pytorch_inference
    DEPENDS ml_test_pytorch_inference
    COMMAND ml_test_pytorch_inference
    WORKING_DIRECTORY ../../bin/pytorch_inference/unittest
)

