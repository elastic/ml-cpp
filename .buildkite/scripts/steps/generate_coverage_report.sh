#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

set -eo pipefail

find ./cmake-build-docker -name "*.gcda" -print0 | xargs -0 -n 1 -P $(nproc) gcov --preserve-paths

# Crate gcov.tar.gz from all the .gcov files in .
find . -name "*.gcov" -print0 | tar -czf gcov.tar.gz --null -T -
rm *.gcov
