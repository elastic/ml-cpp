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

FROM ubuntu:20.04

# This is basically automating the setup instructions in build-setup/macos_cross_compiled.md

MAINTAINER David Roberts <dave.roberts@elastic.co>

# Make sure apt-get is up to date and required packages are installed
RUN \
  export DEBIAN_FRONTEND=noninteractive && \
  apt-get update && \
  apt-get install --no-install-recommends -y apt-utils automake autogen build-essential bzip2 git gobjc gpg-agent libtool software-properties-common unzip wget zip

# Install clang
RUN \
  wget --quiet -O - http://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
  apt-add-repository "deb http://apt.llvm.org/focal/ llvm-toolchain-focal main" && \
  apt-get install --no-install-recommends -y clang-8 libclang1-8 libllvm8 llvm-8 llvm-8-runtime

# Add build dependencies transferred from native Mac build server
RUN \
  mkdir -p /usr/local/sysroot-x86_64-apple-macosx10.14/usr && \
  cd /usr/local/sysroot-x86_64-apple-macosx10.14/usr && \
  wget --quiet -O - https://s3-eu-west-1.amazonaws.com/prelert-artifacts/dependencies/usr-x86_64-apple-macosx10.14-6.tar.bz2 | tar jxf - && \
  wget --quiet -O - https://s3-eu-west-1.amazonaws.com/prelert-artifacts/dependencies/xcode-x86_64-apple-macosx10.14-1.tar.bz2 | tar jxf - && \
  wget --quiet -O - https://s3-eu-west-1.amazonaws.com/prelert-artifacts/dependencies/sdk-x86_64-apple-macosx10.14-1.tar.bz2 | tar jxf -

# Build cctools-port
RUN \
  git clone https://github.com/tpoechtrager/cctools-port.git && \
  cd cctools-port/cctools && \
  git checkout 949.0.1-ld64-530 && \
  export CC=clang-8 && \
  export CXX=clang++-8 && \
  ./autogen.sh && \
  ./configure --target=x86_64-apple-macosx10.14 --with-llvm-config=/usr/bin/llvm-config-8 && \
  make -j`nproc` && \
  make install && \
  cd ../.. && \
  rm -rf cctools-port

# Install CMake
# v3.19.2 minimum is required
RUN \
  wget --quiet https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2-Linux-x86_64.sh && \
  chmod +x cmake-3.23.2-Linux-x86_64.sh && \
  ./cmake-3.23.2-Linux-x86_64.sh --skip-license --prefix=/usr/local && \
  rm -f cmake-3.23.2-Linux-x86_64.sh

