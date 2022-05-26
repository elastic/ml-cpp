#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
endif()

add_subdirectory(autodetect)
add_subdirectory(categorize)
add_subdirectory(controller)
add_subdirectory(data_frame_analyzer)
add_subdirectory(normalize)
add_subdirectory(pytorch_inference)
