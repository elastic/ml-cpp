#!/bin/bash
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

# Shared helpers for Buildkite pipeline scripts that trigger elasticsearch-serverless
# (deploy QA, es-pr-check). Source after setting ML_CPP_ROOT, then call:
#   prepareMlCppServerlessTriggerContext "${BASH_SOURCE[0]}" || exit 1
#   assignServerlessQaTriggerEnvYamlEscapes
# Use $(emitServerlessUploadMlCppDepsStepYaml) inside a "steps:" heredoc body.

function yamlDoubleQuoteEscape {
    printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

function assignServerlessQaTriggerEnvYamlEscapes {
    KEEP_DEPLOYMENT_SAFE=$(yamlDoubleQuoteEscape "${KEEP_DEPLOYMENT:-false}")
    REGION_ID_SAFE=$(yamlDoubleQuoteEscape "${REGION_ID:-aws-eu-west-1}")
    PROJECT_TYPE_SAFE=$(yamlDoubleQuoteEscape "${PROJECT_TYPE:-elasticsearch}")
}

function prepareMlCppServerlessTriggerContext {
    local pipeline_script="$1"
    if [ -z "$pipeline_script" ]; then
        echo "prepareMlCppServerlessTriggerContext: missing path to the calling pipeline script" >&2
        return 1
    fi

    SAFE_MESSAGE=$(printf '%s' "${BUILDKITE_MESSAGE}" | head -1 | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
    PR_NUM="${BUILDKITE_PULL_REQUEST}"
    if [ -z "${PR_NUM}" ] || [ "${PR_NUM}" = "false" ]; then
        PR_NUM="manual"
    fi

    PR_AUTHOR_FORK="$(expr "${BUILDKITE_BRANCH:-}" : '\(.*\):.*' 2>/dev/null || true)"
    PR_SOURCE="$(expr "${BUILDKITE_BRANCH:-}" : '.*:\(.*\)' 2>/dev/null || true)"
    PR_TARGET="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"

    ML_CPP_ROOT="$(cd "$(dirname "${pipeline_script}")/../.." && pwd)"
    # shellcheck source=dev-tools/pick_elasticsearch_clone_target.sh
    source "${ML_CPP_ROOT}/dev-tools/pick_elasticsearch_clone_target.sh"
    export PR_AUTHOR="${PR_AUTHOR_FORK}"
    export PR_SOURCE_BRANCH="${PR_SOURCE}"
    export PR_TARGET_BRANCH="${PR_TARGET}"

    # shellcheck source=dev-tools/pick_elasticsearch_serverless_branch.sh
    source "${ML_CPP_ROOT}/dev-tools/pick_elasticsearch_serverless_branch.sh"
    pickElasticsearchServerlessBranch || return 1

    pickCloneTarget || true
    ES_COMMIT="$(elasticsearch_selected_branch_head_sha)"
    ES_COMMIT="${ES_COMMIT:-HEAD}"
    echo "Resolved elasticsearch submodule: ${SELECTED_FORK}/${SELECTED_BRANCH} -> ${ES_COMMIT}" >&2
}

function emitServerlessUploadMlCppDepsStepYaml {
    cat <<'EOS'
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

EOS
}

