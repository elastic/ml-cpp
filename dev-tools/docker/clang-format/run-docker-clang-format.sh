#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

docker run --rm -v $CPP_SRC_HOME:/ml-cpp -u $(id -u):$(id -g) alpine:clang-format
