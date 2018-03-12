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

# Ensure $CPP_SRC_HOME is set
if [ -z "$CPP_SRC_HOME" ] ; then
    echo '$CPP_SRC_HOME is not set'
    exit 1
fi

# Ensure astyle is available

which astyle > /dev/null 2>&1

if [ $? != 0 ] ; then
    echo "ERROR: The astyle code formatter is not available. Exiting."
    exit 1;
fi

function format() 
{
  perl -0777 -pe 's/\n\s*(\|\||&&)/ \1\n/g' ${1} | astyle;
}

function format_in_place() 
{
    format $1 > $1.new
    mv $1.new $1
}

export -f format
export -f format_in_place
