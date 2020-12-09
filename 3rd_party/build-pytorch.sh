#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
# Build the pytorch libraries. The Git submodules must have been updated before
# building. This Command need only be run once
#  git submodule update --init --recursive

mkdir pytorch/build_libtorch && cd pytorch/build_libtorch
python ../tools/build_libtorch.py