#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#

set -eo pipefail

export CWD="$(pwd)"

# Install the sonar-scanner
cd /usr/local
curl -O https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-6.1.0.4477-linux-x64.zip
unzip sonar-scanner-cli-6.1.0.4477-linux-x64.zip
mv sonar-scanner-6.1.0.4477-linux-x64/ sonar-scanner
export PATH=$(pwd):$PATH

# Generate the compile_commands.json file

cd "${CWD}"

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -B cmake-build-docker

# Analyze the source code with SonarQube

trap 'error_handler $?' EXIT

retries=0

slack_notification() {
  local err
  err=$1

  if [[ "${BUILDKITE}" == "true" ]]; then
    echo "Sending Slack notification that Sonar Scanner step failed with exit code ${err}. Please retry the other step and not this one."
    cat <<-HEREDOC | buildkite-agent pipeline upload
    steps:
      - label: ":slack: Slack Notification - Sonar Scanner Failed"
        command: "echo 'Sonar Scanner failed with exit code ${err}'"
        soft_fail: false
        agents:
          image: "docker.elastic.co/ci-agent-images/basic-buildkite-agent"
        notify:
          - slack:
              channels:
                - "#sonar"
              message: "Sonar Scanner failed with exit code ${err} for attempt ${retries} on ${BUILDKITE_PIPELINE_SLUG}"
HEREDOC
  else
    echo "Sonar Scanner failed with exit code ${err}"
  fi
}

error_handler() {
  local err
  err=$1

  # Since we need to trap the sonar-scanner script exit code, we don't generate notifications on success.
  if [[ $err -eq 0 ]]; then
    exit "$err"
  fi
  
  slack_notification "$err"

  exit "$err"
}

runScanner(){
  local max_retries=5
  local retry_delay=30
  local exit_code=-1 # Initialize to a negative value to allow while loop to run.

  while [[ $retries -lt $max_retries ]] && [[ $exit_code -ne 0 ]]; do
    set +e  # Temporarily disable exit immediately on error to allow retries
    if [[ ${BUILDKITE_PULL_REQUEST} =~ ^[0-9]+$ ]];
    then
      echo "Spotted PR"
      sonar-scanner -Dsonar.token="${SONAR_LOGIN}" \
                    -Dsonar.pullrequest.key="${BUILDKITE_PULL_REQUEST}" \
                    -Dsonar.pullrequest.branch="${BUILDKITE_BRANCH}" \
                    -Dsonar.pullrequest.base="${BUILDKITE_PULL_REQUEST_BASE_BRANCH}" \
                    -Dsonar.projectVersion="${BUILDKITE_COMMIT}" \
                    -Dsonar.scm.provider=git 
      exit_code=$?
    else
      sonar-scanner -Dsonar.token="${SONAR_LOGIN}"  \
                    -Dsonar.branch.name="${BUILDKITE_BRANCH}" \
                    -Dsonar.projectVersion="${BUILDKITE_COMMIT}" \
                    -Dsonar.scm.provider=git 
      exit_code=$?
    fi
    set -e # Re-enable exit immediately on error
 
    if [[ $exit_code -ne 0 ]]; then
      retries=$((retries + 1))
      echo "Sonar Scanner failed with exit code ${exit_code}. Retrying in ${retry_delay} seconds..."
      # Only send slack notification on first and last retry to avoid spamming the channel.
      if [[ $retries -eq 1 ]] || [[ $retries -eq $max_retries ]]; then
        slack_notification "${exit_code}"
      fi
      sleep $retry_delay
      retry_delay=$((retry_delay * retries))
    fi
  done

  return "$exit_code"
}

# Check if we are in a git repo
git rev-parse --is-inside-work-tree >/dev/null

# SonarQube project analyse token was provided
if [[ -z "${SONAR_LOGIN}" ]]; then
  echo "No SONAR_LOGIN token was provided, attempting to resolve it via vault..."

  if [[ -z "${VAULT_ADDR}" ]];
  then
      echo "VAULT_ADDR is missing."
      exit 1
  fi
  if [[ -z "${VAULT_TOKEN}" ]];
  then
      echo "A VAULT_TOKEN is missing for ${VAULT_ADDR}."
      exit 1
  fi
  if [[ -z "${VAULT_SONAR_TOKEN_PATH}" ]];
  then
      echo "VAULT_SONAR_TOKEN_PATH is missing."
      exit 1
  fi

  if [[ "$VAULT_SONAR_TOKEN_PATH" =~ ^kv/* ]];
  then
    SONAR_LOGIN=$(vault kv get --field token "${VAULT_SONAR_TOKEN_PATH}")
  else
    SONAR_LOGIN=$(vault read --field token "${VAULT_SONAR_TOKEN_PATH}")
  fi
fi

runScanner

# cat cmake-build-docker/compile_commands.json | sed "s|$(pwd)|.|g" > compile_commands.json.tmp && mv compile_commands.json.tmp cmake-build-docker/compile_commands.json
# sed -i "s|/usr/local/gcc103/bin/g++|g++|g" cmake-build-docker/compile_commands.json
