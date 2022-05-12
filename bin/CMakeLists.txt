if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
endif()

if(MSVC)
  set(VCINCLUDES "-I${LOCAL_DRIVE}/$ENV{VCBASE}/VC/Tools/MSVC/$ENV{VCVER}/include")
  set(WINSDKINCLUDES "-I${LOCAL_DRIVE}/$ENV{WINSDKBASE}/10/Include/$ENV{UCRTVER}/ucrt -I${LOCAL_DRIVE}/$ENV{WINSDKBASE}/8.0/Include/shared -I${LOCAL_DRIVE}/$ENV{WINSDKBASE}/8.0/Include/um")
  set(CPPFLAGS "-X -I${CMAKE_SOURCE_DIR}/3rd_party/include -I${LOCAL_DRIVE}/usr/local/include ${VCINCLUDES} ${WINSDKINCLUDES} -D${OS} -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE -DWIN32_LEAN_AND_MEAN -DNTDDI_VERSION=0x06010000 -D_WIN32_WINNT=0x0601 -DBUILDING_autodetect ${OPTCPPFLAGS}")
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

