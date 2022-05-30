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
if(CPP_CROSS_COMPILE)
  message(STATUS "Cross compiling: CPP_CROSS_COMPILE = ${CPP_CROSS_COMPILE}")

  set(CROSS_FLAGS --sysroot=${SYSROOT} -B /usr/local/bin -target ${CROSS_TARGET_PLATFORM} -stdlib=libc++)
  set(ML_SHARED_LINKER_FLAGS ${CROSS_FLAGS})
  set(ML_EXE_LINKER_FLAGS    ${CROSS_FLAGS})

  # which compilers to use for C and C++
  set(CMAKE_C_COMPILER   "clang-8")
  set(CMAKE_CXX_COMPILER "clang++-8")
  
  set(CMAKE_AR  "/usr/local/bin/${CROSS_TARGET_PLATFORM}-ar")
  set(CMAKE_RANLIB  "/usr/local/bin/${CROSS_TARGET_PLATFORM}-ranlib")
  set(CMAKE_STRIP  "/usr/local/bin/${CROSS_TARGET_PLATFORM}-strip")
  set(CMAKE_LD  "/usr/local/bin/${CROSS_TARGET_PLATFORM}-ld")
  
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> -ru <TARGET> <OBJECTS>")
  
  # where is the target environment located
  set(CMAKE_FIND_ROOT_PATH  /usr/local/sysroot-${CROSS_TARGET_PLATFORM})
else()
  set(CMAKE_C_COMPILER   "clang")
  set(CMAKE_CXX_COMPILER "clang++")
  set(CMAKE_AR  "ar")
  set(CMAKE_RANLIB  "ranlib")
  set(CMAKE_STRIP  "strip")
  
  #set(Boost_COMPILER "-clang-darwin13")
endif()


list(APPEND ML_C_FLAGS 
    "-g"
    ${CROSS_FLAGS}
    ${OPTCFLAGS}
    ${ARCHCFLAGS}
    "-fstack-protector"
    "-Weverything"
    "-Werror-switch"
    "-Wno-deprecated"
    "-Wno-disabled-macro-expansion"
    "-Wno-documentation-deprecated-sync"
    "-Wno-documentation-unknown-command"
    "-Wno-extra-semi-stmt"
    "-Wno-float-equal"
    "-Wno-missing-prototypes"
    "-Wno-padded"
    "-Wno-poison-system-directories"
    "-Wno-sign-conversion"
    "-Wno-unknown-warning-option"
    "-Wno-unreachable-code"
    "-Wno-used-but-marked-unused"
    ${ML_COVERAGE})

list(APPEND ML_CXX_FLAGS 
    ${ML_C_FLAGS}
    "-Wno-c++98-compat"
    "-Wno-c++98-compat-pedantic"
    "-Wno-exit-time-destructors"
    "-Wno-global-constructors"
    "-Wno-return-std-move-in-c++11"
    "-Wno-unused-member-function"
    "-Wno-weak-vtables")

