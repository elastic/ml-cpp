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

set(REQUIRE "requirements.txt")
set(SYS_VIRTUEALENV "python3 -m venv")
set(ENV "env")
set(BIN "${ENV}/bin")
set(PIP "${BIN}/pip")
set(TEST_DIR tests)
set(PYTHON_EXECUTABLE "${BIN}/python")
set(TEST_RUNNER "${BIN}/py.test")
set(BUMPVERSION "bumpversion")

set(TEST_NOTEBOOKS
    incremental_training
)
set(PACKAGES
    incremental_learning 
)

# TODO: Do I need to add requirements/target dependencies?
add_test(NAME python-tests
    COMMAND ${TEST_RUNNER} ${TEST_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)


