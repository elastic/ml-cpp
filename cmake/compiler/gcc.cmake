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
if(CROSS_TARGET_PLATFORM  STREQUAL "aarch64-linux-gnu")
  set(CMAKE_C_COMPILER   "/usr/local/gcc103/bin/${CROSS_TARGET_PLATFORM}-gcc")
  set(CMAKE_CXX_COMPILER "/usr/local/gcc103/bin/${CROSS_TARGET_PLATFORM}-g++")

  set(CMAKE_SYSTEM_PROCESSOR "aarch64")
  set(CMAKE_AR       "/usr/local/gcc103/bin/${CROSS_TARGET_PLATFORM}-ar")
  set(CMAKE_RANLIB   "/usr/local/gcc103/bin/${CROSS_TARGET_PLATFORM}-ranlib")
  set(CMAKE_STRIP    "/usr/local/gcc103/bin/${CROSS_TARGET_PLATFORM}-strip")
  set(CMAKE_LINKER   "/usr/local/gcc103/bin/${CROSS_TARGET_PLATFORM}-ld")

  message(STATUS "CROSS_TARGET_PLATFORM=${CROSS_TARGET_PLATFORM}")
  message(STATUS "CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD}")
  set(SYSROOT /usr/local/sysroot-${CROSS_TARGET_PLATFORM})
  set(CROSS_FLAGS "--sysroot=${SYSROOT}")

else()
  set(CMAKE_C_COMPILER   "/usr/local/gcc103/bin/gcc")
  set(CMAKE_CXX_COMPILER "/usr/local/gcc103/bin/g++")

  set(CMAKE_SYSTEM_PROCESSOR "x86_64")
  set(CMAKE_AR       "/usr/local/gcc103/bin/ar")
  set(CMAKE_RANLIB   "/usr/local/gcc103/bin/ranlib")
  set(CMAKE_STRIP    "/usr/local/gcc103/bin/strip")
  set(CMAKE_LINKER   "/usr/local/gcc103/bin/ld")
endif()

SET(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> -ru <TARGET> <OBJECTS>")

list(APPEND ML_C_FLAGS
  ${CROSS_FLAGS}
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


