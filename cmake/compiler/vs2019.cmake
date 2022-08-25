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

# TODO: when everything else that needs to be consistent is ready, set this to the current drive instead of hard coding C:
set(ROOT "C:")
if(DEFINED ENV{ROOT})
  set(ROOT $ENV{ROOT})
endif()

set(PFX86_DIR "${ROOT}/Program Files (x86)")
set(MSVC_DIR "${PFX86_DIR}/Microsoft Visual Studio")
set(VCBASE "${MSVC_DIR}/2019/Professional")
set(WINSDKBASE "${PFX86_DIR}/Windows Kits")
set(WIN_KITS_DIR "${PFX86_DIR}/Windows Kits")

file(GLOB MSVC_VERS "${VCBASE}/VC/Tools/MSVC/*")
list(GET MSVC_VERS -1 MSVC_VER)
if(${MSVC_VER} MATCHES "/([^/]+)$")
  set(VCVER ${CMAKE_MATCH_1})
endif()

file(GLOB WINSDK_VERS "${WINSDKBASE}/10/Include/*")
list(GET WINSDK_VERS -1 WINSDK_VER)
message(STATUS "WINSDK_VER ${WINSDK_VER}")
if(${WINSDK_VER} MATCHES "/([^/]+)$")
  set(UCRTVER ${CMAKE_MATCH_1})
endif()

message(STATUS "VCBASE ${VCBASE}")
message(STATUS "WINSDKBASE ${WINSDKBASE}")
message(STATUS "VCVER ${VCVER}")
message(STATUS "UCRTVER ${UCRTVER}")

STRING(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" VCVER_REGEX_MATCH  ${VCVER})
set(VCVER_MAJOR ${CMAKE_MATCH_1})
set(VCVER_MINOR ${CMAKE_MATCH_2})

include("${CMAKE_CURRENT_LIST_DIR}/msvc.cmake")
