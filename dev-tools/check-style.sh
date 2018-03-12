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

. `dirname $0`/astyle_utils.sh

infiles=`git diff --name-only --diff-filter=ACMRT | grep -v "3rd_party" | grep -E "\.(cc|h)$"`

unset ARTISTIC_STYLE_PROJECT_OPTIONS
export ARTISTIC_STYLE_OPTIONS=$CPP_SRC_HOME/.astyle_options

astyle_files=()

for file in ${infiles}; do
  fqfile=`git rev-parse --show-toplevel`/${file}
  echo "Checking: ${file}"
  if ! cmp -s ${fqfile} <(format ${fqfile}); then
    astyle_files+=("${file}")
  fi
done

rc=0

if [ -n "${astyle_files}" ]; then
echo "Astyle has detected a Format error within the following files:"
printf "%s\n" "${astyle_files[@]}"
rc=1
fi

exit ${rc}
