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
# temporary access key and secret key for accessing AWS:
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

# Temporary installation of aws cli to test the credentials obtained from vault
mkdir ~/bin
mkdir ~/aws-cli
wget "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -O "awscliv2.zip"
jar -xvf awscliv2.zip
chmod u+x aws/install
chmod u+x aws/dist/aws
./aws/install -i ~/aws-cli -b ~/bin
export PATH=~/bin:$PATH

unset ML_AWS_ACCESS_KEY ML_AWS_SECRET_KEY ML_AWS_SECURITY_TOKEN
FAILURES=0
while [[ $FAILURES -lt 3 && -z "$ML_AWS_ACCESS_KEY" ]] ; do
    echo "vault read -format=json -field=data aws-elastic-ci-prod/creds/prelert-artifacts"
    AWS_CREDS=$(vault read -format=json -field=data aws-elastic-ci-prod/creds/prelert-artifacts)
    if [ $? -eq 0 ] ; then
        echo "Successfully read creds from vault"
        echo "================================="
        echo $AWS_CREDS
        echo "================================="

        echo "Parsing credentials"
        export AWS_ACCESS_KEY_ID=$(echo $AWS_CREDS | jq -r '.access_key')
        export AWS_SECRET_ACCESS_KEY=$(echo $AWS_CREDS | jq -r '.secret_key')
        export AWS_SESSION_TOKEN=$(echo $AWS_CREDS | jq -r '.security_token')

        export ML_AWS_ACCESS_KEY=$AWS_ACCESS_KEY_ID
        export ML_AWS_SECRET_KEY=$AWS_SECRET_ACCESS_KEY

        env

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

mkdir ~/.aws
cat <<EOL > ~/.aws/credentials
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
aws_session_token = $AWS_SESSION_TOKEN
EOL

cat <<EOL > ~/.aws/config
[default]
region = us-east-1
EOL

echo "listing aws s3 bucket"
aws s3 ls prelert-artifacts/maven/org/elasticsearch/ml/ml-cpp/ || echo "aws ls command failed.. continuing anyway"


if [ "$REENABLE_X_OPTION" = true ] ; then
    set -x
fi
