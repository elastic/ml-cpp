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

ML_CPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=dev-tools/serverless_buildkite_trigger_prepare.sh
source "${ML_CPP_ROOT}/dev-tools/serverless_buildkite_trigger_prepare.sh"

prepareMlCppServerlessTriggerContext "${BASH_SOURCE[0]}" || exit 1
assignServerlessQaTriggerEnvYamlEscapes

echo "Deploying to serverless QA with custom ml-cpp from PR #${PR_NUM}" >&2

cat <<EOL
steps:
$(emitServerlessUploadMlCppDepsStepYaml)
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
        KEEP_DEPLOYMENT: "${KEEP_DEPLOYMENT_SAFE}"
        REGION_ID: "${REGION_ID_SAFE}"
        PROJECT_TYPE: "${PROJECT_TYPE_SAFE}"
EOL
