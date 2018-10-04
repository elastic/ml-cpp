#!/usr/bin/env bash

: "${HOME:?Need to set HOME to a non-empty value.}"
: "${WORKSPACE:?Need to set WORKSPACE to a non-empty value.}"

set +x
export VAULT_TOKEN=$(vault write -field=token auth/approle/login role_id="$VAULT_ROLE_ID" secret_id="$VAULT_SECRET_ID")

aws_creds=$(vault read -format=json -field=data aws-dev/creds/prelertartifacts)
export ML_AWS_ACCESS_KEY=$(echo $aws_creds | jq -r '.access_key')
export ML_AWS_SECRET_KEY=$(echo $aws_creds | jq -r '.secret_key')

unset VAULT_TOKEN VAULT_ROLE_ID VAULT_SECRET_ID
set -ex

# Build and test
./dev-tools/ci

