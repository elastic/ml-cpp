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

# Builds and pushes the CentOS 7 build images with the compromised eMudhra roots
# distrusted, as thin layers over the existing images (see
# linux_distrust_overlay/Dockerfile):
#
#   ml-linux-build:32a                -> ml-linux-build:32b
#   ml-linux-aarch64-cross-build:15a  -> ml-linux-aarch64-cross-build:15b
#
# The toolchain is not rebuilt, so this takes minutes.  This script is not
# intended to be run regularly.  It never overwrites an existing tag.

if [ `uname -m` != x86_64 ] ; then
    echo "Native build images must be built on the correct hardware architecture"
    echo "Required: x86_64, Current:" `uname -m`
    exit 1
fi

HOST=docker.elastic.co
ACCOUNT=ml-dev
# REPOSITORY:BASE_VERSION:NEW_VERSION
IMAGES="ml-linux-build:32a:32b ml-linux-aarch64-cross-build:15a:15b"

set -e

cd `dirname $0`

CONTEXT=linux_distrust_overlay
SCRIPT=../../.buildkite/scripts/common/distrust_emudhra_cas.sh
trap "rm -f $CONTEXT/distrust_emudhra_cas.sh" EXIT

for image in $IMAGES; do
    REPOSITORY=`echo $image | cut -d: -f1`
    BASE=$HOST/$ACCOUNT/$REPOSITORY:`echo $image | cut -d: -f2`
    NEW=$HOST/$ACCOUNT/$REPOSITORY:`echo $image | cut -d: -f3`
    if docker manifest inspect $NEW > /dev/null 2>&1 ; then
        echo "$NEW already exists: not overwriting it"
        exit 1
    fi
    docker pull $BASE
    cp $SCRIPT $CONTEXT/distrust_emudhra_cas.sh
    docker build --no-cache --build-arg BASE_IMAGE=$BASE -t $NEW $CONTEXT
done

# Get a username and password for this by visiting
# https://docker-auth.elastic.co and allowing it to authenticate against your
# GitHub account
docker login $HOST
for image in $IMAGES; do
    docker push $HOST/$ACCOUNT/`echo $image | cut -d: -f1`:`echo $image | cut -d: -f3`
done
