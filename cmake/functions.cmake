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

  file(READ ${CMAKE_SOURCE_DIR}/gradle.properties GRADLE_PROPERTIES)
  if(${GRADLE_PROPERTIES} MATCHES "elasticsearchVersion=([0-9.]+)")
    set(ML_VERSION_STR "${CMAKE_MATCH_1}")
  endif()

  if(ENV{VERSION_QUALIFIER})
    set(ML_VERSION_STR "${ML_VERSION_STR}-$ENV{VERSION_QUALIFIER}")
  endif()

  if(NOT ENV{SNAPSHOT} STREQUAL no)
    set(ML_VERSION_STR "${ML_VERSION_STR}-SNAPSHOT")
  endif()

  if(${ML_VERSION_STR} MATCHES "([0-9.]+)")
    set(ML_VERSION ${CMAKE_MATCH_1})
  endif()
  string(REPLACE "." "," ML_VERSION "${ML_VERSION}")

  set(ML_PATCH "0")

  execute_process(COMMAND git -c core.fileMode=false update-index -q --refresh ERROR_FILE /dev/null OUTPUT_FILE /dev/null)
  execute_process(COMMAND git -c core.fileMode=false diff-index --quiet HEAD --  RESULT_VARIABLE UNCOMMITTED_CHANGES)

  if(UNCOMMITED_CHANGES EQUAL 0)
    set(ML_FILEFLAGS "0")
  else()
    set(ML_FILEFLAGS "VS_FF_PRIVATEBUILD")
  endif()

  if("${_target}" MATCHES ".dll$")
    set(ML_FILETYPE "VFT_DLL")
  else()
    set(ML_FILETYPE "VFT_APP")
  endif()

  set(ML_FILENAME ${_target})

  if(${_target} MATCHES "([^.]+).")
    set(ML_NAME ${CMAKE_MATCH_1})
  endif()
  set(ML_YEAR ${BUILD_YEAR})
  set(ML_ICON ${CMAKE_SOURCE_DIR}/mk/ml.ico)

  configure_file(
	"${CMAKE_SOURCE_DIR}/mk/ml.rc.in"
	"${CMAKE_CURRENT_BINARY_DIR}/${ML_NAME}.rc"
        @ONLY
        )

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

  if (WIN32 AND _type STREQUAL "SHARED")
    ml_generate_resources(lib${_target}.dll)
    list(APPEND PLATFORM_SRCS ${CMAKE_CURRENT_BINARY_DIR}/lib${_target}.rc)
  endif()

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

  if (WIN32)
    ml_generate_resources(${_target}.exe)
    list(APPEND PLATFORM_SRCS ${CMAKE_CURRENT_BINARY_DIR}/${_target}.rc)
  endif()

  add_executable(${_target} Main.cc ${PLATFORM_SRCS})

  if (ML_EXE_LINKER_FLAGS)
    target_link_options(${_target} PUBLIC ${ML_EXE_LINKER_FLAGS})
  endif()

  target_link_libraries(${_target} PUBLIC ${ML_LINK_LIBRARIES})

  ml_install(${_target})

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
        COMMAND ${CMAKE_COMMAND} -DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR} -DTEST_NAME=ml_test_${_target} -P ${CMAKE_SOURCE_DIR}/cmake/test-runner.cmake
      COMMENT "Running test: ml_test_${_target}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      )
  else()
    add_custom_target(test_${_target}
      DEPENDS ml_test_${_target}
      COMMAND ${CMAKE_COMMAND} -DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR} -DTEST_NAME=ml_test_${_target} -P ${CMAKE_SOURCE_DIR}/cmake/test-runner.cmake
      COMMENT "Running test: ml_test_${_target}"
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


#
# Add a "doxygen" target to generate Doxygen documentation
# from the source files. Returns false with an error message
# if the doxygen executable cannot be found.
#
function(ml_doxygen _output)
  find_program(DOXYGEN_EXECUTABLE doxygen HINTS /opt/homebrew /usr/local)
  find_package(Doxygen)
  if (NOT DOXYGEN_FOUND)
    add_custom_target(doxygen COMMAND false
                              COMMENT "Doxygen not found")
    return()
  endif()

  set(DOXYGEN_GENERATE_HTML          YES)
  set(DOXYGEN_HTML_OUTPUT            ${_output})
  set(DOXYGEN_PROJECT_NAME           "ML C++")
  set(DOXYGEN_PROJECT_NUMBER         ${ML_VERSION_NUM})
  set(DOXYGEN_PROJECT_LOGO           mk/ml.ico)
  set(DOXYGEN_OUTPUT_DIRECTORY       ${CMAKE_SOURCE_DIR}/build/doxygen)
  set(DOXYGEN_INHERIT_DOCS           NO)
  set(DOXYGEN_SEPARATE_MEMBER_PAGES  NO)
  set(DOXYGEN_TAB_SIZE               4)
  set(DOXYGEN_LOOKUP_CACHE_SIZE      1)
  set(DOXYGEN_EXTRACT_ALL            YES)
  set(DOXYGEN_EXTRACT_PRIVATE        YES)
  set(DOXYGEN_EXTRACT_STATIC         YES)
  set(DOXYGEN_EXTRACT_ANON_NSPACES   YES)
  set(DOXYGEN_FILE_PATTERNS          *.cc  *.h)
  set(DOXYGEN_RECURSIVE              YES)
  set(DOXYGEN_EXCLUDE                3rd_party)
  set(DOXYGEN_HTML_OUTPUT            cplusplus)
  set(DOXYGEN_SEARCHENGINE           NO)
  set(DOXYGEN_PAPER_TYPE             a4wide)
  set(DOXYGEN_EXTRA_PACKAGES         amsmath amssymb)
  set(DOXYGEN_LATEX_BATCHMODE        YES)
  set(DOXYGEN_HAVE_DOT               YES)
  set(DOXYGEN_DOT_FONTNAME           FreeSans)
  set(DOXYGEN_DOT_GRAPH_MAX_NODES    100)

  doxygen_add_docs(doxygen
      ${PROJECT_SOURCE_DIR}
      COMMENT "Generate HTML documentation"
  )

set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${_output})

endfunction()
