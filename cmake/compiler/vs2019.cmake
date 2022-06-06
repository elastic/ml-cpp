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

if (DEFINED VS2019_VARIABLES_)
    return ()
else()
    set (VS2019_VARIABLES_ 1)
endif()

set(ROOT "/c")
if(DEFINED ENV{ROOT})
  set(ROOT $ENV{ROOT})
endif()
  
execute_process(COMMAND bash -c "cygpath -m -s \"${ROOT}/Program Files (x86)/Microsoft Visual Studio/2019/Professional\"" OUTPUT_VARIABLE VCBASE OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND bash -c "cygpath -m -s \"${ROOT}/Program Files (x86)/Windows Kits\"" OUTPUT_VARIABLE WINSDKBASE OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND bash -c "cygpath -m -s \"${ROOT}/Program Files (x86)\"" OUTPUT_VARIABLE PFX86_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND bash -c "cd ${PFX86_DIR} && cygpath -m -s \"Microsoft Visual Studio\"" OUTPUT_VARIABLE MSVC_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND bash -c "cd ${PFX86_DIR} && cygpath -m -s \"Windows Kits\"" OUTPUT_VARIABLE WIN_KITS_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND bash -c "/bin/ls -1 ${PFX86_DIR}/${MSVC_DIR}/2019/Professional/VC/Tools/MSVC" COMMAND bash -c "tail -1" OUTPUT_VARIABLE VCVER OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND bash -c "/bin/ls -1 ${WINSDKBASE}/10/Include" COMMAND bash -c "tail -1" OUTPUT_VARIABLE UCRTVER OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "VCVER ${VCVER}")
STRING(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" VCVER_REGEX_MATCH  ${VCVER})
set(VCVER_MAJOR ${CMAKE_MATCH_1})
set(VCVER_MINOR ${CMAKE_MATCH_2})

include("${CMAKE_CURRENT_LIST_DIR}/msvc.cmake")
