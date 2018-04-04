#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Reformats Ml native source code, using clang-format,  to ensure consistency.

# Ensure $CPP_SRC_HOME is set
if [ -z "$CPP_SRC_HOME" ] ; then
    echo '$CPP_SRC_HOME is not set'
    exit 1
fi

find $CPP_SRC_HOME \( -name 3rd_party -o -name build-setup \) -prune -o \( -name \*.cc -o -name \*.h \) -exec clang-format -i {} \;
