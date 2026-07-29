#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

SAFE_MESSAGE=$(printf '%s' "${BUILDKITE_MESSAGE}" | head -1 | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')

# Derive STACK_VERSION / ES_BRANCH so release-branch and backport builds test
# against the matching stack version and ES branch instead of the qaf-tests
# defaults (main / current-dev SNAPSHOT). Silent on stdout by contract.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/derive_qa_stack_env.sh"

cat <<EOL
steps:
  - label: "Trigger Appex PyTorch Tests :test_tube:"
    command:
      - echo 'Trigger PyTorch Tests'
      - 'buildkite-agent artifact download "build/*" . --step build_test_linux-x86_64-RelWithDebInfo'
    depends_on: "test_linux-x86_64-RelWithDebInfo"
    notify:
      -  github_commit_status:
           context: "Trigger Appex QA PyTorch Tests"
  - wait
  - trigger: appex-qa-stateful-custom-ml-cpp-build-testing
    async: false
    build:
      message: "${SAFE_MESSAGE}"
      env:
        QAF_TESTS_TO_RUN: "pytorch_tests"
EOL

if [ "${ES_BRANCH}" != "" ]; then
cat <<EOL
        ES_BRANCH: "${ES_BRANCH}"
EOL
fi

if [ "${STACK_VERSION}" != "" ]; then
cat <<EOL
        STACK_VERSION: "${STACK_VERSION}"
EOL
fi
