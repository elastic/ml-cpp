#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# RUN_RSS_PROBE (issue #3226): when set, the downstream pr-build runs the
# standalone pytorch_inference OOM repro (run_rss_probe action) instead of the
# full QA suite. Buildkite downstream builds do not inherit env, so the action
# and any RSS_PROBE_* overrides are threaded explicitly through the trigger.
DOWNSTREAM_ACTION="run_pytorch_tests"
if [ "${RUN_RSS_PROBE:-}" = "true" ]; then
    DOWNSTREAM_ACTION="run_rss_probe"
fi

RSS_PROBE_ENV_LINES=""
for probe_var in RSS_PROBE_NUM_REQUESTS RSS_PROBE_BATCH_SIZE RSS_PROBE_NUM_TOKENS \
                 RSS_PROBE_THREADS_PER_ALLOCATION RSS_PROBE_ALLOCATIONS \
                 RSS_PROBE_MAX_SECONDS RSS_PROBE_INPUT_MODE RSS_PROBE_MODEL_URL \
                 RSS_PROBE_DIST_BUILD RSS_PROBE_DIST_PIPELINE; do
    probe_val="${!probe_var:-}"
    if [ -n "${probe_val}" ]; then
        RSS_PROBE_ENV_LINES="${RSS_PROBE_ENV_LINES}
        ${probe_var}: \"${probe_val}\""
    fi
done

echo "---"
echo "steps:"

# In prebuilt-probe mode we reuse an already-built distribution downstream, so
# there is no need to (re)build the libtorch Docker image here.
if [ -z "${RSS_PROBE_DIST_BUILD:-}" ]; then
cat <<EOL
  - label: "Build PyTorch Docker Image"
    key: "build_pytorch_docker_image"
    command: "./dev-tools/docker/build_pytorch_linux_build_image.sh"
    agents:
      "provider": "gcp"
      "machineType": "c2-standard-16"
    notify:
      - github_commit_status:
          context: "Build PyTorch Docker image"
  - wait
EOL
fi

cat <<EOL
  - trigger: ml-cpp-pr-builds
    async: false
    build:
      branch: "${BUILDKITE_BRANCH}"
      commit: "${BUILDKITE_COMMIT}"
      message: "${BUILDKITE_MESSAGE}"
      env:
        DOCKER_IMAGE: "docker.elastic.co/ml-dev/ml-linux-dependency-build:pytorch_latest"
        GITHUB_PR_COMMENT_VAR_PLATFORM: "linux"
        GITHUB_PR_COMMENT_VAR_ARCH: "x86_64"
        GITHUB_PR_COMMENT_VAR_ACTION: "${DOWNSTREAM_ACTION}"
        GITHUB_PR_TRIGGER_COMMENT: ""${RSS_PROBE_ENV_LINES}
EOL
