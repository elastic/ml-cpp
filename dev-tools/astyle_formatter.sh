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

. `dirname $0`/astyle_utils.sh

export ARTISTIC_STYLE_OPTIONS=$CPP_SRC_HOME/.astyle_options

# astyle does have some limitations, particularly around where it decides to align continuation lines
# To alleviate some of the issues we first run a perl script to ensure that the boolean operators || and &&
# always appear at the end of a line, never the beginning.
find $CPP_SRC_HOME -name 3rd_party -prune -o \( -name \*.cc -o -name \*.h \) -exec bash -c "format_in_place $0" {} \;
