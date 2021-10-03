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

# Run a FOSSA scan on the repo to check for license violations

# Get the FOSSA API token from Vault
set +x
export VAULT_TOKEN=$(vault write -field=token auth/approle/login role_id="$VAULT_ROLE_ID" secret_id="$VAULT_SECRET_ID")
unset VAULT_ROLE_ID VAULT_SECRET_ID
export FOSSA_API_KEY=$(vault read -field=token secret/jenkins-ci/fossa/api-token)
unset VAULT_TOKEN
set -x

# Change directory to the top level of the repo
readonly GIT_TOPLEVEL=$(git rev-parse --show-toplevel 2> /dev/null)
cd "$GIT_TOPLEVEL"

# Run the FOSSA scan
fossa analyze

