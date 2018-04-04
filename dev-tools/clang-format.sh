#!/bin/bash
#
# ELASTICSEARCH CONFIDENTIAL
#
# Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
#
# Notice: this software, and all information contained
# therein, is the exclusive property of Elasticsearch BV
# and its licensors, if any, and is protected under applicable
# domestic and foreign law, and international treaties.
#
# Reproduction, republication or distribution without the
# express written consent of Elasticsearch BV is
# strictly prohibited.
#

# Reformats Ml native source code, using clang-format,  to ensure consistency.

# Ensure $CPP_SRC_HOME is set
if [ -z "$CPP_SRC_HOME" ] ; then
    echo '$CPP_SRC_HOME is not set'
    exit 1
fi

find $CPP_SRC_HOME \( -name 3rd_party -o -name build-setup \) -prune -o \( -name \*.cc -o -name \*.h \) -exec clang-format -i {} \;
