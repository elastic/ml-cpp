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

message(STATUS "x86_64 detected for target")
#set(CMAKE_SYSTEM_PROCESSOR "x86_64")
set (ARCHCFLAGS "-msse4.2")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    string(APPEND ARCHCFLAGS " -mfpmath=sse")
endif()
add_compile_definitions(RAPIDJSON_SSE42)

