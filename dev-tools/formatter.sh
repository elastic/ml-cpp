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

. `dirname $0`/format_utils.sh

find $CPP_SRC_HOME -name 3rd_party -prune -o \( -name \*.cc -o -name \*.h \) -exec bash -c "format_in_place {}" \;
