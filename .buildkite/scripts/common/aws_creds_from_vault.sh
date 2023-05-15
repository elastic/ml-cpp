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
# temporary access key, secret key and session token for accessing AWS:
# - ML_AWS_ACCESS_KEY
# - ML_AWS_SECRET_KEY
# - ML_AWS_SESSION_TOKEN
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

unset ML_AWS_ACCESS_KEY ML_AWS_SECRET_KEY ML_AWS_SESSION_TOKEN
FAILURES=0
while [[ $FAILURES -lt 3 && -z "$ML_AWS_ACCESS_KEY" ]] ; do
    AWS_CREDS=$(vault read -format=json -field=data aws-elastic-ci-prod/creds/prelert-artifacts)
    if [ $? -eq 0 ] ; then
        export ML_AWS_ACCESS_KEY=$(echo $AWS_CREDS | jq -r '.access_key')
        export ML_AWS_SECRET_KEY=$(echo $AWS_CREDS | jq -r '.secret_key')
        export ML_AWS_SESSION_TOKEN=$(echo $AWS_CREDS | jq -r '.security_token')
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
