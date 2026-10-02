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
PROBE_NUM_REQUESTS="${RSS_PROBE_NUM_REQUESTS:-3000}"
PROBE_BATCH_SIZE="${RSS_PROBE_BATCH_SIZE:-16}"
PROBE_NUM_TOKENS="${RSS_PROBE_NUM_TOKENS:-512}"
PROBE_THREADS="${RSS_PROBE_THREADS_PER_ALLOCATION:-4}"
PROBE_ALLOCATIONS="${RSS_PROBE_ALLOCATIONS:-1}"
PROBE_MAX_SECONDS="${RSS_PROBE_MAX_SECONDS:-3600}"
PROBE_INPUT_MODE="${RSS_PROBE_INPUT_MODE:-file}"
ELSER_URL="${RSS_PROBE_MODEL_URL:-https://ml-models.elastic.co/elser_model_2_linux-x86_64.pt}"

# Distribution source: prebuilt (cross-build) vs this pipeline's build step.
if [ -n "${RSS_PROBE_DIST_BUILD:-}" ]; then
    DEPENDS_BLOCK=""
    DOWNLOAD_CMD="buildkite-agent artifact download \"build/distributions/ml-cpp-*-SNAPSHOT-linux-x86_64.zip\" . --build \"${RSS_PROBE_DIST_BUILD}\""
    DIST_NOTE="prebuilt distribution from build ${RSS_PROBE_DIST_BUILD}"
else
    DEPENDS_BLOCK="    depends_on: \"build_test_linux-x86_64-RelWithDebInfo\""
    DOWNLOAD_CMD="buildkite-agent artifact download \"build/distributions/ml-cpp-*-SNAPSHOT-linux-x86_64.zip\" . --step build_test_linux-x86_64-RelWithDebInfo"
    DIST_NOTE="distribution from this pipeline's x86_64 build"
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
      RSS_PROBE_NUM_REQUESTS: "${PROBE_NUM_REQUESTS}"
      RSS_PROBE_BATCH_SIZE: "${PROBE_BATCH_SIZE}"
      RSS_PROBE_NUM_TOKENS: "${PROBE_NUM_TOKENS}"
      RSS_PROBE_THREADS_PER_ALLOCATION: "${PROBE_THREADS}"
      RSS_PROBE_ALLOCATIONS: "${PROBE_ALLOCATIONS}"
      RSS_PROBE_MAX_SECONDS: "${PROBE_MAX_SECONDS}"
      RSS_PROBE_INPUT_MODE: "${PROBE_INPUT_MODE}"
      RSS_PROBE_MODEL_URL: "${ELSER_URL}"
    command: |
      set -euo pipefail
      echo "--- Obtaining linux-x86_64 distribution (${DIST_NOTE})"
      ${DOWNLOAD_CMD}
      DIST_ZIP=\$(find build/distributions -name "ml-cpp-*-SNAPSHOT-linux-x86_64.zip" ! -name "*debug*" | head -1)
      echo "distribution: \${DIST_ZIP}"
      mkdir -p dist && (cd dist && unzip -q -o "../\${DIST_ZIP}")
      PYTORCH_BIN=\$(find dist -type f -name pytorch_inference | head -1)
      echo "pytorch_inference: \${PYTORCH_BIN}"
      LIB_DIR=\$(dirname \$(dirname "\${PYTORCH_BIN}"))/lib
      export LD_LIBRARY_PATH="\${LIB_DIR}:\${LD_LIBRARY_PATH:-}"
      echo "--- Downloading ELSER model"
      curl -fL --retry 5 --retry-delay 5 -o elser.pt "\${RSS_PROBE_MODEL_URL}" || wget -O elser.pt "\${RSS_PROBE_MODEL_URL}"
      ls -la elser.pt
      echo "--- Running RSS probe"
      python3 dev-tools/pytorch_inference_rss_probe.py --app "\${PYTORCH_BIN}" --model elser.pt --num-requests "\${RSS_PROBE_NUM_REQUESTS}" --batch-size "\${RSS_PROBE_BATCH_SIZE}" --num-tokens "\${RSS_PROBE_NUM_TOKENS}" --num-threads-per-allocation "\${RSS_PROBE_THREADS_PER_ALLOCATION}" --num-allocations "\${RSS_PROBE_ALLOCATIONS}" --max-seconds "\${RSS_PROBE_MAX_SECONDS}" --input-mode "\${RSS_PROBE_INPUT_MODE}" --label "\${BUILDKITE_BRANCH}" --csv rss_probe.csv | tee rss_probe.log
      buildkite-agent artifact upload "rss_probe.csv"
      buildkite-agent artifact upload "rss_probe.log"
    notify:
      - github_commit_status:
          context: "RSS probe (pytorch_inference OOM)"
EOL
