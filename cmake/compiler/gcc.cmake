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

# Set CODE_COVERAGE if it is defined in the environment
if(DEFINED ENV{CODE_COVERAGE})
  set(CODE_COVERAGE $ENV{CODE_COVERAGE})
endif()

option(CODE_COVERAGE "Enable code coverage" OFF)

if(CODE_COVERAGE)
  message(STATUS "Code coverage enabled.")
  set(COVERAGE_CXX_FLAGS "-fprofile-arcs -ftest-coverage")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COVERAGE_CXX_FLAGS}")
#  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${COVERAGE_CXX_FLAGS}")
#  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${COVERAGE_CXX_FLAGS}")
#  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${COVERAGE_CXX_FLAGS}")
#  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COVERAGE_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgcov")
else()
  message(STATUS "Code coverage disabled.")
endif()

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


