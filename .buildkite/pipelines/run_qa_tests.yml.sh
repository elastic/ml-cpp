#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Derive STACK_VERSION / ES_BRANCH so release-branch and backport builds test
# against the matching stack version and ES branch instead of the qaf-tests
# defaults (main / current-dev SNAPSHOT). Any values already set (e.g. from PR
# comment vars) are preserved. Silent on stdout by contract.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/derive_qa_stack_env.sh"

# Derive a human-readable descriptor for the trigger step's label and GitHub
# commit-status context from the requested suites. This single script now
# handles QA-only, PyTorch-only and combined runs, so a fixed "QA Tests" label
# would mislabel the other two. Substring matches (rather than exact tokens)
# keep this correct for marker expressions such as "ml_cpp_pr and not slow".
QAF_SUITES="${QAF_TESTS_TO_RUN:-ml_cpp_pr}"
_has_qa=false
_has_pytorch=false
case "${QAF_SUITES}" in *ml_cpp_pr*) _has_qa=true ;; esac
case "${QAF_SUITES}" in *pytorch_tests*) _has_pytorch=true ;; esac
if [ "${_has_qa}" = true ] && [ "${_has_pytorch}" = true ]; then
    QA_TESTS_DESC="QA + PyTorch"
elif [ "${_has_pytorch}" = true ]; then
    QA_TESTS_DESC="PyTorch"
else
    QA_TESTS_DESC="QA"
fi

cat <<EOL
steps:
  - label: "Trigger Appex ${QA_TESTS_DESC} Tests :test_tube:"
    command:
      - echo 'Trigger ${QA_TESTS_DESC} Tests'
      - 'buildkite-agent artifact download "build/*" . --step build_test_linux-x86_64-RelWithDebInfo'
    depends_on: "build_test_linux-x86_64-RelWithDebInfo"
    notify:
      -  github_commit_status:
           context: "Trigger Appex ${QA_TESTS_DESC} Tests"
  - wait
  - trigger: appex-qa-stateful-custom-ml-cpp-build-testing
    async: false
    build:
      message: "${BUILDKITE_MESSAGE}"
      env:
        QAF_TESTS_TO_RUN: "${QAF_TESTS_TO_RUN:-ml_cpp_pr}"
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
