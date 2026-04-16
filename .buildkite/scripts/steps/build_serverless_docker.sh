#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Builds a serverless Elasticsearch Docker image that incorporates custom
# ml-cpp artifacts from the current PR build.
#
# Prerequisites:
#   - ml-cpp build artifacts in build/distributions/ml-cpp-*-linux-x86_64.zip
#     (downloaded from a prior Buildkite step)
#
# Environment:
#   IMAGE_TAG                - Docker image tag (required)
#   ES_SERVERLESS_BRANCH     - elasticsearch-serverless branch to build from (default: main)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ES_SERVERLESS_BRANCH="${ES_SERVERLESS_BRANCH:-main}"
DOCKER_REGISTRY="docker.elastic.co/elasticsearch-ci/elasticsearch-serverless"

VERSION=$(grep '^elasticsearchVersion' "${REPO_ROOT}/gradle.properties" | awk -F= '{ print $2 }' | xargs echo)
VERSION="${VERSION}-SNAPSHOT"

echo "--- Setting up local Ivy repo with custom ml-cpp ${VERSION}"
IVY_REPO="$(pwd)/ivy-repo"
IVY_ML_DIR="${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/${VERSION}"
mkdir -p "$IVY_ML_DIR"

cp "build/distributions/ml-cpp-${VERSION}-linux-x86_64.zip" \
   "${IVY_ML_DIR}/ml-cpp-${VERSION}.zip"
cp "build/distributions/ml-cpp-${VERSION}-linux-x86_64.zip" \
   "${IVY_ML_DIR}/ml-cpp-${VERSION}-nodeps.zip"
cp "${REPO_ROOT}/dev-tools/minimal.zip" \
   "${IVY_ML_DIR}/ml-cpp-${VERSION}-deps.zip"

IVY_REPO_URL="file://${IVY_REPO}"
echo "Ivy repo URL: ${IVY_REPO_URL}"

echo "--- Obtaining GitHub token for elasticsearch-serverless"
set +x
# ES_SERVERLESS_GITHUB_TOKEN can be provided as a build env var when the
# pipeline's default VAULT_GITHUB_TOKEN lacks access to elasticsearch-serverless.
# Long-term, the ml-cpp pipeline's Vault role should be granted read access to
# a GitHub App token that covers elasticsearch-serverless.
ES_SERVERLESS_TOKEN="${ES_SERVERLESS_GITHUB_TOKEN:-${VAULT_GITHUB_TOKEN:-}}"
if [ -z "${ES_SERVERLESS_TOKEN}" ]; then
  echo "ERROR: Could not obtain a GitHub token with access to elasticsearch-serverless."
  echo "Set ES_SERVERLESS_GITHUB_TOKEN as a build environment variable."
  exit 1
fi
set -x

echo "--- Cloning elasticsearch-serverless (branch: ${ES_SERVERLESS_BRANCH})"
cd ..
rm -rf elasticsearch-serverless
set +x
git clone --depth=1 -b "${ES_SERVERLESS_BRANCH}" \
    "https://x-access-token:${ES_SERVERLESS_TOKEN}@github.com/elastic/elasticsearch-serverless.git"
set -x
cd elasticsearch-serverless

echo "--- Initializing elasticsearch submodule"
set +x
git config url."https://x-access-token:${ES_SERVERLESS_TOKEN}@github.com/".insteadOf "git@github.com:"
set -x
git submodule update --init --depth=1

echo "--- Building serverless Docker image with custom ml-cpp"
# The serverless build requires a license key for release builds
LICENSE_KEY=$(mktemp -d)/license.key
vault read -field pubkey \
    secret/ci/elastic-elasticsearch-serverless/migrated/es-license \
    | base64 --decode > "$LICENSE_KEY"

./gradlew --console=plain --parallel \
    -Dbuild.snapshot=false \
    "-Dlicense.key=${LICENSE_KEY}" \
    "-Dbuild.ml_cpp.repo=${IVY_REPO_URL}" \
    buildDockerImage

echo "--- Tagging and pushing Docker image"
FULL_TAG="${DOCKER_REGISTRY}:${IMAGE_TAG}"
docker tag elasticsearch-serverless:x86_64 "${FULL_TAG}"

set +x
DOCKER_REGISTRY_USERNAME="$(vault read -field=username secret/ci/elastic-elasticsearch-serverless/prod_docker_registry_credentials)"
DOCKER_REGISTRY_PASSWORD="$(vault read -field=password secret/ci/elastic-elasticsearch-serverless/prod_docker_registry_credentials)"
echo "${DOCKER_REGISTRY_PASSWORD}" | docker login -u "${DOCKER_REGISTRY_USERNAME}" --password-stdin docker.elastic.co
set -x

docker push "${FULL_TAG}"
echo "Pushed ${FULL_TAG}"

# Store the image tag as metadata for downstream steps
buildkite-agent meta-data set "serverless-image-tag" "${IMAGE_TAG}"
buildkite-agent meta-data set "serverless-image" "${FULL_TAG}"
