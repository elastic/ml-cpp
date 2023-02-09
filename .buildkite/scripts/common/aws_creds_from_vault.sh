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

# Gets AWS access credentials from Vault.
#
# Designed to be run by sourcing into the script that requires the credentials.
# Requires the jq utility.  Disables command tracing during execution to
# prevent sensitive information getting into the console output.
#
# On success the following environment variables will be set that contain the
# temporary # access key and secret key for accessing AWS:
# - ML_AWS_ACCESS_KEY
# - ML_AWS_SECRET_KEY
#
# On failure this script will exit, so will terminate the script that sourced
# it.
#
# It is pointless to run this script in a sub-process - it must be sourced by
# some other script to be of any use.

case $- in
    *x*)
        set +x
        REENABLE_X_OPTION=true
        ;;
    *)
        REENABLE_X_OPTION=false
        ;;
esac

# We obtain the AWS credentials in two stages:
#
# 1. Obtain the role and secret id necessary to access https://secrets.elastic.co:8200
#    from where they've been stored in CI vault

# Variables named *_PASSWORD, *_SECRET, *_TOKEN, *_ACCESS_KEY & *_SECRET_KEY are redacted in BuildKite’s environment
# so store the role and secret id in them for security
VAULT_ACCESS_KEY=`vault read -field=role_id secret/ci/elastic-ml-cpp/aws-dev/creds/prelertartifacts`
VAULT_SECRET_KEY=`vault read -field=secret_id secret/ci/elastic-ml-cpp/aws-dev/creds/prelertartifacts`

#
# 2. Use the role and secret id obtained above to access the AWS secrets engine in https://secrets.elastic.co:8200
#    and query it for the AWS access and secret keys.
#
export VAULT_TOKEN=$(VAULT_ADDR=https://secrets.elastic.co:8200 vault write -field=token auth/approle/login role_id="$VAULT_ACCESS_KEY" secret_id="$VAULT_SECRET_KEY")

unset ML_AWS_ACCESS_KEY ML_AWS_SECRET_KEY
FAILURES=0
while [[ $FAILURES -lt 3 && -z "$ML_AWS_ACCESS_KEY" ]] ; do
    AWS_CREDS=$(VAULT_ADDR=https://secrets.elastic.co:8200 vault read -format=json -field=data aws-dev/creds/prelertartifacts)
    if [ $? -eq 0 ] ; then
        export ML_AWS_ACCESS_KEY=$(echo $AWS_CREDS | jq -r '.access_key')
        export ML_AWS_SECRET_KEY=$(echo $AWS_CREDS | jq -r '.secret_key')
    fi
    if [ -z "$ML_AWS_ACCESS_KEY" ] ; then
        let FAILURES++
        echo "Attempt $FAILURES to get AWS credentials failed"
    fi
done

if [ -z "$ML_AWS_ACCESS_KEY" -o -z "$ML_AWS_SECRET_KEY" ] ; then
    echo "Exiting after failing to get AWS credentials $FAILURES times"
    exit 1
fi

if [ "$REENABLE_X_OPTION" = true ] ; then
    set -x
fi
