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

# Reformats Ml native code (*.cc, *.h files) to ensure a consistent style

# Ensure astyle is available

which astyle > /dev/null 2>&1

if [ $? != 0 ] ; then
    echo "ERROR: The astyle code formatter is not available. Exiting."
    exit 1;
fi

# run astyle with the following options:
# A2: Java style (attached) braces
# C: indent classes
# S: indent switches
# O: keep one-line blocks intact
# M100: Set the maximum number of spaces to indent continuation lines to be 100
# r "*.cc,*.h": Operate on *.cc and *.h files recursively...
# --exclude=3rd_party: but ignore files under the 3rd_party directory
# n: Do not make any backups of the original files.
# Note: The default indentation level is 4 spaces & the default continuation indent is 1 level
astyle -A2 -C -S -O -M100 -r "*.cc,*.h" --exclude=3rd_party -n
