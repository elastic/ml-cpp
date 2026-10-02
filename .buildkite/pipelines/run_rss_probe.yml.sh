#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Emits a single step that reproduces the pytorch_inference OOM (issue #3226)
# WITHOUT Elasticsearch: it obtains a linux-x86_64 distribution (pytorch_inference
# + bundled libtorch), fetches the ELSER model from the public model repository,
# and drives a sustained inference load through the standalone process while
# sampling RSS. This collapses the ~6h nightly QA cycle into a direct memory
# trace in minutes.
#
# Two sources for the distribution:
#   * default: the x86_64 build step of THIS pipeline run (depends_on), or
#   * prebuilt (RSS_PROBE_DIST_BUILD set): a previously-built distribution from
#     another ml-cpp-pr-builds build, identified by its build UUID/number. This
#     lets us probe an exact ml-cpp+libtorch combination that was already built
#     (e.g. b0c52aa4 + the Oct-1 bb63328c libtorch from A/B Arm B) with no rebuild.
#
# Launched via the run_rss_probe action (see ml_pipeline/config.py). All
# RSS_PROBE_* knobs are overridable from the triggering build's environment.

# Image: reuse the image the x86_64 build ran in (has libtorch runtime deps and
# python3); fall back to the standard linux build image. In prebuilt mode the
# distribution bundles its own libtorch, so the image only provides python3/curl.
PROBE_IMAGE="${DOCKER_IMAGE:-docker.elastic.co/ml-dev/ml-linux-build:34}"

# Probe parameters (overridable via the triggering build environment).
# Defaults chase the QA OOM: a large, variable-length (wikipedia-like) request
# stream across many parallel allocations, which exercises many distinct tensor
# shapes and is the workload that drives unbounded RSS growth.
PROBE_NUM_REQUESTS="${RSS_PROBE_NUM_REQUESTS:-20000}"
PROBE_BATCH_SIZE="${RSS_PROBE_BATCH_SIZE:-16}"
PROBE_NUM_TOKENS="${RSS_PROBE_NUM_TOKENS:-512}"
PROBE_VARY_TOKENS="${RSS_PROBE_VARY_TOKENS:-true}"
PROBE_MIN_TOKENS="${RSS_PROBE_MIN_TOKENS:-1}"
PROBE_THREADS="${RSS_PROBE_THREADS_PER_ALLOCATION:-4}"
PROBE_ALLOCATIONS="${RSS_PROBE_ALLOCATIONS:-8}"
PROBE_MAX_SECONDS="${RSS_PROBE_MAX_SECONDS:-3600}"
PROBE_INPUT_MODE="${RSS_PROBE_INPUT_MODE:-file}"
ELSER_URL="${RSS_PROBE_MODEL_URL:-https://ml-models.elastic.co/elser_model_2_linux-x86_64.pt}"

# Distribution source: prebuilt (cross-build) vs this pipeline's build step.
# The probe body lives in a committed script (dev-tools/run_rss_probe_ci.sh) and
# is driven purely by the env: block below. We deliberately keep ALL shell
# variables out of the YAML command, because `buildkite-agent pipeline upload`
# interpolates ${VAR} in the pipeline at upload time and would blank out values
# meant to be resolved at runtime.
if [ -n "${RSS_PROBE_DIST_BUILD:-}" ]; then
    DEPENDS_BLOCK=""
    DIST_BUILD_ENV="      RSS_PROBE_DIST_BUILD: \"${RSS_PROBE_DIST_BUILD}\""
else
    DEPENDS_BLOCK="    depends_on: \"build_test_linux-x86_64-RelWithDebInfo\""
    DIST_BUILD_ENV="      RSS_PROBE_DIST_STEP: \"build_test_linux-x86_64-RelWithDebInfo\""
fi

cat <<EOL
steps:
  - label: "Reproduce pytorch_inference OOM (RSS probe) :chart_with_upwards_trend:"
    key: "rss_probe_linux-x86_64"
${DEPENDS_BLOCK}
    timeout_in_minutes: 120
    agents:
      cpu: "6"
      ephemeralStorage: "20G"
      memory: "64G"
      image: "${PROBE_IMAGE}"
    env:
${DIST_BUILD_ENV}
      RSS_PROBE_NUM_REQUESTS: "${PROBE_NUM_REQUESTS}"
      RSS_PROBE_BATCH_SIZE: "${PROBE_BATCH_SIZE}"
      RSS_PROBE_NUM_TOKENS: "${PROBE_NUM_TOKENS}"
      RSS_PROBE_VARY_TOKENS: "${PROBE_VARY_TOKENS}"
      RSS_PROBE_MIN_TOKENS: "${PROBE_MIN_TOKENS}"
      RSS_PROBE_THREADS_PER_ALLOCATION: "${PROBE_THREADS}"
      RSS_PROBE_ALLOCATIONS: "${PROBE_ALLOCATIONS}"
      RSS_PROBE_MAX_SECONDS: "${PROBE_MAX_SECONDS}"
      RSS_PROBE_INPUT_MODE: "${PROBE_INPUT_MODE}"
      RSS_PROBE_MODEL_URL: "${ELSER_URL}"
    command: "bash dev-tools/run_rss_probe_ci.sh"
    notify:
      - github_commit_status:
          context: "RSS probe (pytorch_inference OOM)"
EOL
