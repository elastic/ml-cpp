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

#
# Generate the additional resource file containing the ML icon and build information
# that is linked into the Windows executables
#
function(ml_generate_resources _target)
  if(NOT WIN32)
    return()
  endif()
  set( ${_target}_LINKFLAGS ${CMAKE_CURRENT_BINARY_DIR}/${_target}.res )
  set_target_properties( ${_target} PROPERTIES LINK_FLAGS ${${_target}_LINKFLAGS} )
  execute_process(COMMAND bash -c "${CMAKE_SOURCE_DIR}/mk/make_rc_defines.sh ${_target}.exe" OUTPUT_VARIABLE
    RC_DEFINES OUTPUT_STRIP_TRAILING_WHITESPACE)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh "rc -nologo ${CPPFLAGS} ${RC_DEFINES} -Fo${_target}.res ${CMAKE_SOURCE_DIR}/mk/ml.rc")
  add_custom_target(
    ${_target}.res
    DEPENDS ${CMAKE_SOURCE_DIR}/mk/ml.rc ${CMAKE_SOURCE_DIR}/gradle.properties ${CMAKE_SOURCE_DIR}/mk/ml.ico ${CMAKE_SOURCE_DIR}/mk/make_rc_defines.sh ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh
    COMMAND bash -c ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh
    )
  add_dependencies(${_target} ${_target}.res)

  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/tmp.sh)
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/${_target}.res)
endfunction()

#
# Generate a list of source files, replacing any in the input list
# with a platform specific file if one exists.
#
function(ml_generate_platform_sources)
  set(SRCS ${ARGN})
  foreach(FILE ${SRCS})
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

#
# Install a target into the correct location
# depending on the type of target and the platform
#
function(ml_install _target)
  install(TARGETS ${_target} 
    LIBRARY DESTINATION ${DYNAMIC_LIB_DIR}
    RUNTIME DESTINATION ${EXE_DIR}
    ARCHIVE DESTINATION ${IMPORT_LIB_DIR}
    )

  if(WIN32)
    set_property(TARGET ${_target} PROPERTY COMPILE_PDB_NAME ${_target})
    install(FILES $<TARGET_PDB_FILE:${_target}> DESTINATION ${DYNAMIC_LIB_DIR} OPTIONAL)
  endif()
endfunction()

#
# Rules to create and install  a library target
# _type may be SHARED or STATIC
#
function(ml_add_library _target _type)
  set(SRCS ${ARGN})

  ml_generate_platform_sources(${SRCS})

  include_directories(${CMAKE_SOURCE_DIR}/include)

  add_compile_definitions(BUILDING_lib${_target})

  add_library(${_target} ${_type} ${PLATFORM_SRCS})

  if(ML_LINK_LIBRARIES)
    target_link_libraries(${_target} PUBLIC ${ML_LINK_LIBRARIES})
  endif()

  if(ML_DEPENDENCIES)
    add_dependencies(${_target} ${ML_DEPENDENCIES})
  endif()

  if (_type STREQUAL "SHARED")
    if (ML_SHARED_LINKER_FLAGS)
      target_link_options(${_target} PUBLIC ${ML_SHARED_LINKER_FLAGS})
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      target_link_libraries(${_target} PRIVATE
        "-current_version ${ML_VERSION_NUM}"
        "-compatibility_version ${ML_VERSION_NUM}"
        "${COVERAGE}")
    endif()

    ml_install(${_target})
  endif()
endfunction()

#
# Rules to create and install an executable target
# Note that 'Main.cc' is included by default so there
# is no neeed to pass it in.
# Also note that a transient OBJECT library
# is created, this simplifies the task of creating a unit
# test that links against this executable target's object
# files
#
function(ml_add_executable _target)
  set(SRCS ${ARGN})

  ml_generate_platform_sources(${SRCS})

  include_directories(${CMAKE_SOURCE_DIR}/include)

  if(PLATFORM_SRCS)
    add_library(Ml${_target} OBJECT ${PLATFORM_SRCS})
  endif()

  add_executable(${_target} Main.cc ${PLATFORM_SRCS})

  if (ML_EXE_LINKER_FLAGS)
    target_link_options(${_target} PUBLIC ${ML_EXE_LINKER_FLAGS})
  endif()

  target_link_libraries(${_target} PUBLIC ${ML_LINK_LIBRARIES})

  ml_install(${_target})

  ml_generate_resources(${_target})

  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(${_target} PRIVATE "${COVERAGE}")
  endif()
endfunction()

#
# Create a unit test executable and rules for running it.
# The input parameter '_target' corresponds to either 
# a library or executable target and will have 'ml_test_' prepended
# to it to form the name of the unit test executable.
# If an OBJECT library exists corresponding to the name of the
# input parameter '_target' with 'Ml' prepended then the
# constituent object files from that library will be linked
# in to the unit test executable.
#
function(ml_add_test_executable _target)
  set(SRCS ${ARGN})

  ml_generate_platform_sources(${SRCS})

  include_directories(${CMAKE_SOURCE_DIR}/include)

  add_executable(ml_test_${_target} EXCLUDE_FROM_ALL  ${PLATFORM_SRCS}
    $<$<TARGET_EXISTS:Ml${_target}>:$<TARGET_OBJECTS:Ml${_target}>>)

  target_link_libraries(ml_test_${_target} ${ML_LINK_LIBRARIES})

  add_test(ml_test_${_target} ml_test_${_target})

  if(MSVC)
    # For Visual Studio builds the build type forms part of the path to the
    # target. As this isn't known until build time a generator expression is
    # required to determine it.
    # Also, as some unittests make assumptions about the directory that the test
    # executable resides in we copy the test executable up a level in the binary
    # source directory.
    add_custom_target(test_${_target}
      DEPENDS ml_test_${_target}
      COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/$<IF:$<CONFIG:Release>,Release,Debug>/ml_test_${_target}.exe
        ${CMAKE_CURRENT_BINARY_DIR}/ml_test_${_target}.exe
      COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ml_test_${_target}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      )
  else()
    add_custom_target(test_${_target}
      DEPENDS ml_test_${_target}
      COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ml_test_${_target}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      )
  endif()
endfunction()

#
# Add a target to the list of unittests to be built and run
# _directory: Relative path to unittest binary directory, e.g. lib/maths/common/unittest
# _target: Name of the unittest target, e.g. maths_common
#
function(ml_add_test _directory _target)
  add_subdirectory(../${_directory} ${_directory})
  list(APPEND ML_BUILD_TEST_DEPENDS ml_test_${_target})
  list(APPEND ML_TEST_DEPENDS test_${_target})
  set(ML_BUILD_TEST_DEPENDS ${ML_BUILD_TEST_DEPENDS} PARENT_SCOPE)
  set(ML_TEST_DEPENDS ${ML_TEST_DEPENDS} PARENT_SCOPE)
endfunction()
