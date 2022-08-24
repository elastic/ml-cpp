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

# CMake 3.19.2 is the minimum version required to support Apple Silicon
cmake_minimum_required(VERSION 3.19.2)

project(ml-python NONE)

enable_testing()

set(REQUIREMENTS "requirements.txt")
set(REQUIREMENTS_LOG ".requirements.log")
set(SYS_VIRTUEALENV "python3 -m venv")
set(SRC_DIR src)
set(TEST_DIR tests)
set(NB_DIR notebooks)
set(TEST_RUNNER "${BIN}/py.test")

# Output debug information
set(CMAKE_VERBOSE_MAKEFILE OFF)

set(TEST_NOTEBOOKS
    incremental_learning
)
set(PACKAGES
    incremental_learning
)

set(Python_FIND_VIRTUALENV "ONLY")

find_package(Python 3.7 COMPONENTS Interpreter Development REQUIRED)
find_program(Docker_EXECUTABLE docker REQUIRED)

if(EXISTS "${REQUIREMENTS_LOG}")
    message("Requirements already installed. Skip this step.")
else()
    message("Installing PyPI requirements.")
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m pip install -r ${REQUIREMENTS}
        OUTPUT_FILE "${REQUIREMENTS_LOG}"
    )
endif()

message("Installing packages from ${SRC_DIR}")
execute_process(
    COMMAND ${Python_EXECUTABLE} -m pip install -e ${SRC_DIR}
)

add_custom_target(bump_major
    COMMAND ${Python_EXECUTABLE} -m bumpversion major
    WORKING_DIRECTORY ${SRC_DIR}
    COMMENT "Increase the major version of the Python packages."
)

add_custom_target(bump_minor
    COMMAND ${Python_EXECUTABLE} -m bumpversion minor
    WORKING_DIRECTORY ${SRC_DIR}
    COMMENT "Increase the minor version of the Python packages."
)

add_custom_target(src_tests
    COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/${TEST_DIR}
    COMMENT "Run Python unit tests."
)

add_custom_target(notebook_tests
    COMMAND ${Python_EXECUTABLE} -m pytest --nbsmoke-run ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/${NB_DIR}/${TEST_NOTEBOOKS}
    COMMENT "Run Jupyter notebook smoke tests."
)

add_custom_target(build_docker
    COMMAND DOCKER_BUILDKIT=1 docker build -t $ENV{IMAGE_NAME} -f ./jupyter/docker/Dockerfile .
    COMMAND echo "New docker image $ENV{IMAGE_NAME} created."
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/..
    COMMENT "Create docker image for the specified in IMAGE_NAME"
)

add_custom_target(run_docker
    COMMAND docker run -p 9999:9999 $ENV{IMAGE_NAME}:latest
    COMMENT "Run docker container for specified IMAGE_NAME"
)

add_custom_target(clean_build
    COMMAND ${CMAKE_COMMAND} -P clean_build.cmake
)
