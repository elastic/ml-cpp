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

# which compilers to use for C and C++
set(CMAKE_C_COMPILER   "/usr/local/gcc133/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/local/gcc133/bin/g++")

# Detect architecture from ARCHCFLAGS (set by architecture/*.cmake files)
if(ARCHCFLAGS MATCHES "-march=armv8")
  set(CMAKE_SYSTEM_PROCESSOR "aarch64")
else()
  set(CMAKE_SYSTEM_PROCESSOR "x86_64")
endif()

set(CMAKE_AR       "/usr/local/gcc133/bin/ar")
set(CMAKE_RANLIB   "/usr/local/gcc133/bin/ranlib")
set(CMAKE_STRIP    "/usr/local/gcc133/bin/strip")
set(CMAKE_LINKER   "/usr/local/gcc133/bin/ld")

SET(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> -ru <TARGET> <OBJECTS>")

list(APPEND ML_C_FLAGS
  ${ARCHCFLAGS}
  "-fstack-protector"
  "-fno-math-errno"
  "-Wall"
  "-Wcast-align"
  "-Wconversion"
  "-Wextra"
  "-Winit-self"
  "-Wno-psabi"
  "-Wparentheses"
  "-Wpointer-arith"
  "-Wswitch-enum"
  ${ML_COVERAGE}
  )

list(APPEND ML_CXX_FLAGS
  ${ML_C_FLAGS}
  "-Wno-ctor-dtor-privacy"
  "-Wno-deprecated-declarations"
  "-Wold-style-cast"
  "-fvisibility-inlines-hidden"
  )


