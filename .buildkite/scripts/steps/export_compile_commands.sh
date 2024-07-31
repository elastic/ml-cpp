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

set -euo pipefail

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -B cmake-build-docker

cat cmake-build-docker/compile_commands.json | sed "s|$(pwd)|.|g" > compile_commands.json.tmp && mv compile_commands.json.tmp cmake-build-docker/compile_commands.json
sed -i "s|/usr/local/gcc103/bin/g++|g++|g" cmake-build-docker/compile_commands.json
