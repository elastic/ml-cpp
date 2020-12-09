#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
# Script to clone a specific branch/tag of Pytorch into the pytorch directory.
# After cloning the submodules in pytorch are updated
# 

cd `dirname "$BASH_SOURCE"`

rm -rf pytorch 
git -c advice.detachedHead=false clone --depth=1 --branch=v1.7.0 git@github.com:pytorch/pytorch.git

cd pytorch
echo "Updating submodules"
git submodule update --init --recursive
