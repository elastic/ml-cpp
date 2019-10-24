#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Used to create the compile_commands.json files required by CLion

make -Bnwk -j7 | grep -v "Finding dependencies of" | compiledb -o $1

