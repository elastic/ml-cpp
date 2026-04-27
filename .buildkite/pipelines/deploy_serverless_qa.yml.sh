#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Pipeline: build a serverless Docker image with custom ml-cpp and deploy it
# to the QA environment for interactive use. Unlike run_serverless_tests.yml.sh,
# this does NOT run E2E tests -- it just gets the environment running so the
# developer can interact with it (deploy models, run queries, kubectl, etc.).
#
# The deployment stays up for 1 hour by default. Set KEEP_DEPLOYMENT=true
# (via the Buildkite UI) to keep it longer. The build annotations will
# contain the URL and encrypted credentials for accessing the deployment.

SAFE_MESSAGE=$(printf '%s' "${BUILDKITE_MESSAGE}" | head -1 | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
PR_NUM="${BUILDKITE_PULL_REQUEST}"
if [ -z "${PR_NUM}" ] || [ "${PR_NUM}" = "false" ]; then
  PR_NUM="manual"
fi

# Extract PR metadata once for reuse.
PR_AUTHOR_FORK="$(expr "${BUILDKITE_BRANCH:-}" : '\(.*\):.*' 2>/dev/null || true)"
PR_SOURCE="$(expr "${BUILDKITE_BRANCH:-}" : '.*:\(.*\)' 2>/dev/null || true)"
PR_TARGET="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"

ML_CPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=dev-tools/pick_elasticsearch_clone_target.sh
source "${ML_CPP_ROOT}/dev-tools/pick_elasticsearch_clone_target.sh"
export PR_AUTHOR="${PR_AUTHOR_FORK}"
export PR_SOURCE_BRANCH="${PR_SOURCE}"
export PR_TARGET_BRANCH="${PR_TARGET}"

# --- Resolve elasticsearch-serverless branch (shared with run_serverless_tests.yml.sh) ---
# shellcheck source=dev-tools/pick_elasticsearch_serverless_branch.sh
source "${ML_CPP_ROOT}/dev-tools/pick_elasticsearch_serverless_branch.sh"
pickElasticsearchServerlessBranch

# --- Resolve ES submodule commit (shared pick_elasticsearch_clone_target.sh) ---
pickCloneTarget || true
ES_COMMIT="$(elasticsearch_selected_branch_head_sha)"
ES_COMMIT="${ES_COMMIT:-HEAD}"
echo "Resolved elasticsearch submodule: ${SELECTED_FORK}/${SELECTED_BRANCH} -> ${ES_COMMIT}" >&2

echo "Deploying to serverless QA with custom ml-cpp from PR #${PR_NUM}" >&2

cat <<EOL
steps:
  - label: ":package: Upload ml-cpp deps artifact"
    key: "upload_ml_cpp_deps"
    command: 'buildkite-agent artifact upload dev-tools/minimal.zip'
    depends_on:
      - "build_test_linux-x86_64-RelWithDebInfo"
      - "build_test_linux-aarch64-RelWithDebInfo"
    agents:
      provider: aws
      instanceType: m6i.xlarge
      imagePrefix: core-amazonlinux-2023
      diskSizeGb: 100
      diskName: '/dev/xvda'

  - label: ":rocket: Deploy custom ml-cpp to serverless QA"
    depends_on: "upload_ml_cpp_deps"
    async: false
    trigger: elasticsearch-serverless-deploy-qa
    build:
      branch: "${SERVERLESS_BRANCH}"
      message: "ml-cpp PR #${PR_NUM}: ${SAFE_MESSAGE}"
      env:
        ML_CPP_BUILD_ID: "${BUILDKITE_BUILD_ID}"
        ELASTICSEARCH_SUBMODULE_COMMIT: "${ES_COMMIT}"
        KEEP_DEPLOYMENT: "${KEEP_DEPLOYMENT:-false}"
        REGION_ID: "${REGION_ID:-aws-eu-west-1}"
        PROJECT_TYPE: "${PROJECT_TYPE:-elasticsearch}"
EOL
