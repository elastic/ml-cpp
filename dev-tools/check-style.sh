#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

infiles=`git diff --name-only --diff-filter=ACMRT | grep -v "3rd_party" | grep -E "\.(cc|h)$"`

files=()

for file in ${infiles}; do
  fqfile=`git rev-parse --show-toplevel`/${file}
  echo "Checking: ${file}"
  if ! cmp -s ${fqfile} <(clang-format ${fqfile}); then
    files+=("${file}")
  fi
done

rc=0

if [ -n "${files}" ]; then
echo "A Format error has been detected within the following files:"
printf "%s\n" "${files[@]}"
rc=1
fi

exit ${rc}
