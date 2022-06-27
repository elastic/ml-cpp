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
  set(CMAKE_INSTALL_RPATH "\$ORIGIN/")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CMAKE_INSTALL_RPATH "@loader_path")
endif()

add_subdirectory(core)
add_subdirectory(maths/common)
add_subdirectory(maths/time_series)
add_subdirectory(maths/analytics)
add_subdirectory(model)
add_subdirectory(api)
add_subdirectory(api/dump_state EXCLUDE_FROM_ALL)
add_subdirectory(test)
add_subdirectory(ver)
add_subdirectory(seccomp)

