#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

FROM ubuntu:16.04

# Docker image containing the correct clang-format for check-style.sh

MAINTAINER David Roberts <dave.roberts@elastic.co>

# Make sure apt-get is up to date and required packages are installed
RUN \
  apt-get update && \
  apt-get install --no-install-recommends -y git wget xz-utils

# Install the version of LLVM mandated for code formatting
RUN \
  wget --quiet -O - http://releases.llvm.org/5.0.1/clang+llvm-5.0.1-x86_64-linux-gnu-ubuntu-16.04.tar.xz | tar Jxf -

# Update path to find clang-format
ENV PATH /clang+llvm-5.0.1-x86_64-linux-gnu-ubuntu-16.04/bin:/usr/bin:/bin:/usr/sbin:/sbin

