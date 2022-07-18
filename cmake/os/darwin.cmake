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

message(STATUS "Darwin detected")

set(EXE_DIR MacOS)
set(CMAKE_MACOSX_RPATH 1)
add_compile_definitions(MacOSX)
set(PLATFORM_NAME "MacOSX")

# Xcode code signs the binary artifacts by default,
# which we don't want as it is invalidated when
# the RPATH is updated upon install. Hence we disable
# code signing here.
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "")
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO")
