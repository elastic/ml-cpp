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

#
# vault-cp
#
# A script to copy Vault secrets from one path to another.
# Where the first path belongs to https://secrets.elastic.co:8200
# and the second to https://vault-ci-prod.elastic.dev.
# It is likely that this script will only need to be run once.
#

set -eo pipefail

# ensure we were given two command line arguments
if [[ $# -ne 2 ]]; then
  echo 'usage: vault-cp SOURCE DEST' >&2
  echo 'e.g.: vault-cp aws-dev/creds/prelertartifacts secret/ci/elastic-ml-cpp/aws-dev/creds/prelertartifacts' >&2
  exit 1
fi

source=$1
dest=$2

# check for dependencies
if ! command -v jq > /dev/null; then
  echo 'vault-cp: required command "jq" was not found' >&2
  exit 1
fi

printf "Please enter your GitHub token for vault: "
read -r token

PROD_VAULT_TOKEN=`vault login -address https://secrets.elastic.co:8200 -token-only -method github token=${token}`
CI_VAULT_TOKEN=`vault login -address https://vault-ci-prod.elastic.dev -token-only -method github token=${token}`

alias vault-prd="VAULT_TOKEN=$PROD_VAULT_TOKEN VAULT_ADDR=https://secrets.elastic.co:8200 vault"
alias vault-ci="VAULT_TOKEN=$CI_VAULT_TOKEN VAULT_ADDR=https://vault-ci-prod.elastic.dev vault"

source_json=$( VAULT_TOKEN=$PROD_VAULT_TOKEN VAULT_ADDR=https://secrets.elastic.co:8200 vault read -format=json "$source")
source_data=$(echo "$source_json" | jq '.data')
[[ -n $DEBUG ]] && printf '%s\n' "$source_data"

if  VAULT_TOKEN=$CI_VAULT_TOKEN VAULT_ADDR=https://vault-ci-prod.elastic.dev vault read "$dest" > /dev/null 2>&1; then
  overwrite='n'
  printf 'Destination "%s" already exists...overwrite? [y/N] ' "$dest"
  read -r overwrite

  # only overwrite if user explicitly confirms
  if [[ ! $overwrite =~ ^[Yy]$ ]]; then
    echo 'vault-cp: copying has been aborted' >&2
    exit 1
  fi
fi

echo "$source_data" | VAULT_TOKEN=$CI_VAULT_TOKEN VAULT_ADDR=https://vault-ci-prod.elastic.dev vault write "$dest" -
