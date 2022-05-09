if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib:\$ORIGIN/")
endif()

add_subdirectory(autodetect)
add_subdirectory(categorize)
add_subdirectory(controller)
add_subdirectory(controller/unittest)
add_subdirectory(data_frame_analyzer)
add_subdirectory(normalize)
add_subdirectory(pytorch_inference)
add_subdirectory(pytorch_inference/unittest)

