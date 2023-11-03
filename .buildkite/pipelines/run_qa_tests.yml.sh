#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

cat <<EOL
steps:
  - label: "Trigger Appex QA Tests :test_tube:"
    command:
      - echo 'Trigger QA Tests'
      - 'buildkite-agent artifact download "build/*" . --step build_test_linux-x86_64-RelWithDebInfo'
    depends_on: "build_test_linux-x86_64-RelWithDebInfo"
  - wait
  - trigger: appex-qa-stateful-custom-ml-c-plus-plus-build-testing
    async: false
    build:
      message: "${BUILDKITE_MESSAGE}"
      env:
        QAF_TESTS_TO_RUN: "test1,test2,test3"
EOL
