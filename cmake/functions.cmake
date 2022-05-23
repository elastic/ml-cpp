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

if (DEFINED ML_FUNCTIONS_)
    return()
else()
    set (ML_FUNCTIONS_ 1)

    set (CMAKE_BASE_ROOT_DIR ${CMAKE_SOURCE_DIR})
    list (FIND CMAKE_MODULE_PATH "${CMAKE_BASE_ROOT_DIR}/cmake" _index)
    if (${_index} EQUAL -1)
        list (APPEND CMAKE_MODULE_PATH "${CMAKE_BASE_ROOT_DIR}/cmake")
    endif()

    set (ML_CMAKE_DIR ${CMAKE_BASE_ROOT_DIR}/cmake)
endif()

function (ml_show)
    foreach (var ${ARGN})
        message ("variable ${var} -- ${${var}}")
    endforeach()
endfunction()

function(gen_platform_srcs SRCS_LIST)
    foreach(FILE ${SRCS_LIST})
      set(FILE "${CMAKE_CURRENT_SOURCE_DIR}/${FILE}")
      string(REPLACE ".cc" "_${PLATFORM_NAME}.cc" FILE_TMP ${FILE})
      if (EXISTS ${FILE_TMP})
          list(APPEND PLATFORM_SRCS ${FILE_TMP})
      else()
          list(APPEND PLATFORM_SRCS ${FILE})
      endif()
  endforeach()
  set(PLATFORM_SRCS ${PLATFORM_SRCS} PARENT_SCOPE)
endfunction()


