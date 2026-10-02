#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# CI body for the pytorch_inference OOM RSS probe (issue #3226). Obtains a
# linux-x86_64 distribution (pytorch_inference + bundled libtorch), fetches the
# ELSER model from the public model repository, and drives a sustained inference
# load through the standalone process while sampling RSS.
#
# All configuration is read from the environment so that the Buildkite step's
# `command:` can be a single literal invocation. This deliberately avoids
# putting any shell variables in the pipeline YAML, because
# `buildkite-agent pipeline upload` interpolates ${VAR} in the YAML at upload
# time and would otherwise blank out values meant to be evaluated at runtime
# (e.g. $PWD, $(find ...), the probe knobs).
#
# Recognised environment variables (with defaults):
#   RSS_PROBE_DIST_BUILD            if set, download the distribution from this
#                                   other ml-cpp-pr-builds build (UUID/number);
#                                   otherwise download from RSS_PROBE_DIST_STEP
#                                   of the current build.
#   RSS_PROBE_DIST_STEP            step key to download the distribution from
#                                   (default build_test_linux-x86_64-RelWithDebInfo)
#   RSS_PROBE_NUM_REQUESTS          default 3000
#   RSS_PROBE_BATCH_SIZE            default 16
#   RSS_PROBE_NUM_TOKENS            default 512
#   RSS_PROBE_THREADS_PER_ALLOCATION default 4
#   RSS_PROBE_ALLOCATIONS           default 1
#   RSS_PROBE_MAX_SECONDS           default 3600
#   RSS_PROBE_INPUT_MODE            default file
#   RSS_PROBE_MODEL_URL             default public ELSER v2 linux-x86_64 model

set -euo pipefail

NUM_REQUESTS="${RSS_PROBE_NUM_REQUESTS:-20000}"
BATCH_SIZE="${RSS_PROBE_BATCH_SIZE:-16}"
NUM_TOKENS="${RSS_PROBE_NUM_TOKENS:-512}"
VARY_TOKENS="${RSS_PROBE_VARY_TOKENS:-true}"
MIN_TOKENS="${RSS_PROBE_MIN_TOKENS:-1}"
TOKEN_SKEW="${RSS_PROBE_TOKEN_SKEW:-uniform}"
THREADS="${RSS_PROBE_THREADS_PER_ALLOCATION:-4}"
ALLOCATIONS="${RSS_PROBE_ALLOCATIONS:-8}"
MAX_SECONDS="${RSS_PROBE_MAX_SECONDS:-3600}"
INPUT_MODE="${RSS_PROBE_INPUT_MODE:-file}"
MODEL_URL="${RSS_PROBE_MODEL_URL:-https://ml-models.elastic.co/elser_model_2_linux-x86_64.pt}"
DIST_STEP="${RSS_PROBE_DIST_STEP:-build_test_linux-x86_64-RelWithDebInfo}"

VARY_FLAG=""
case "${VARY_TOKENS}" in
    1|true|TRUE|yes|on) VARY_FLAG="--vary-tokens" ;;
esac

REPO_ROOT="${PWD}"
DL_DIR="${REPO_ROOT}/_dist_dl"
DIST_DIR="${REPO_ROOT}/_dist"
rm -rf "${DL_DIR}" "${DIST_DIR}"
mkdir -p "${DL_DIR}" "${DIST_DIR}"

ARTIFACT_GLOB="build/distributions/ml-cpp-*-SNAPSHOT-linux-x86_64.zip"

if [ -n "${RSS_PROBE_DIST_BUILD:-}" ]; then
    echo "--- Obtaining linux-x86_64 distribution (prebuilt from build ${RSS_PROBE_DIST_BUILD})"
    ( cd "${DL_DIR}" && buildkite-agent artifact download "${ARTIFACT_GLOB}" . --build "${RSS_PROBE_DIST_BUILD}" )
else
    echo "--- Obtaining linux-x86_64 distribution (from step ${DIST_STEP} of this build)"
    ( cd "${DL_DIR}" && buildkite-agent artifact download "${ARTIFACT_GLOB}" . --step "${DIST_STEP}" )
fi

DIST_ZIP=$(find "${DL_DIR}" -type f -name "ml-cpp-*-SNAPSHOT-linux-x86_64.zip" ! -name "*debug*" | head -1)
echo "distribution: ${DIST_ZIP}"
if [ -z "${DIST_ZIP}" ]; then
    echo "ERROR: distribution zip not found after download; contents of ${DL_DIR}:"
    find "${DL_DIR}" -maxdepth 4 -type f | head -50
    exit 1
fi

unzip -q -o "${DIST_ZIP}" -d "${DIST_DIR}"
PYTORCH_BIN=$(find "${DIST_DIR}" -type f -name pytorch_inference | head -1)
echo "pytorch_inference: ${PYTORCH_BIN}"
if [ -z "${PYTORCH_BIN}" ]; then
    echo "ERROR: pytorch_inference binary not found in distribution; contents of ${DIST_DIR}:"
    find "${DIST_DIR}" -maxdepth 4 -type f | head -50
    exit 1
fi

LIB_DIR="$(dirname "$(dirname "${PYTORCH_BIN}")")/lib"
export LD_LIBRARY_PATH="${LIB_DIR}:${LD_LIBRARY_PATH:-}"

echo "--- Downloading ELSER model"
curl -fL --retry 5 --retry-delay 5 -o elser.pt "${MODEL_URL}" || wget -O elser.pt "${MODEL_URL}"
ls -la elser.pt

echo "--- Running RSS probe"
python3 dev-tools/pytorch_inference_rss_probe.py \
    --app "${PYTORCH_BIN}" \
    --model elser.pt \
    --num-requests "${NUM_REQUESTS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-tokens "${NUM_TOKENS}" \
    ${VARY_FLAG} \
    --min-tokens "${MIN_TOKENS}" \
    --token-skew "${TOKEN_SKEW}" \
    --num-threads-per-allocation "${THREADS}" \
    --num-allocations "${ALLOCATIONS}" \
    --max-seconds "${MAX_SECONDS}" \
    --input-mode "${INPUT_MODE}" \
    --label "${BUILDKITE_BRANCH:-local}" \
    --csv rss_probe.csv | tee rss_probe.log

buildkite-agent artifact upload "rss_probe.csv"
buildkite-agent artifact upload "rss_probe.log"
