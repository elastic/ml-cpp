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

. `dirname $0`/format_utils.sh

infiles=`git diff --name-only --diff-filter=ACMRT | grep -v "3rd_party" | grep -E "\.(cc|h)$"`

files=()

for file in ${infiles}; do
  fqfile=`git rev-parse --show-toplevel`/${file}
  echo "Checking: ${file}"
  if ! cmp -s ${fqfile} <(format ${fqfile}); then
    files+=("${file}")
  fi
done

rc=0

if [ -n "${files}" ]; then
echo "Format error detected within the following files:"
printf "%s\n" "${files[@]}"
rc=1
fi

exit ${rc}
