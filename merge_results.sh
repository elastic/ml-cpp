#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#

# This script amalgamates multiple JUNIT results files from individual tests
# into one, omitting any test cases that have been skipped. The result is
# output to stdout.

if [ $# -lt 1 ]; then
  echo "Usage: $0 <boost unit test junit results to merge>"
  exit 1
fi

JUNIT_FILES="$@"

echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
cat $JUNIT_FILES | \
  gawk -n '
  BEGIN{tests=0; skipped=0; errors=0; failures=0; id=""; time=0.0; name=""}
  {
    where=match($0, /<testsuite tests="([^"]+)" skipped="([^"]+)" errors="([^"]+)" failures="([^"]+)" id="([^"]+)" name="([a-zA-Z.]+)" time="([^"]+)"/, a)
    if (where != 0) {
      tests+=a[1]; skipped+=a[2]; errors+=a[3]; failures+=a[4]; id=a[5]; name=a[6]; time+=a[7]
  }
  }
  END{print "<testsuite", "tests=\""tests"\"", "skipped=\"0\"", "errors=\""errors"\"", "failures=\""failures"\"", "id=\""id"\"", "name=\""name"\"", "time=\""time"\"" ">"}'

cat $JUNIT_FILES | sed -e '/xml/d' -e '/testsuite/d' -e '/<testcase/i\
' -e '/<\/testcase/a\
' | sed -e '/<testcase/,/testcase>/{H;d;};x;/skipped/d' | grep '.'
echo "</testsuite>"
echo

