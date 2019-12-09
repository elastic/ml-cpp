#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

FROM ubuntu:18.04

# This is basically automating the setup instructions in build-setup/macos_cross_compiled.md

MAINTAINER David Roberts <dave.roberts@elastic.co>

# Make sure apt-get is up to date and required packages are installed
RUN \
  apt-get update && \
  apt-get install --no-install-recommends -y apt-utils automake autogen build-essential bzip2 git gobjc gpg-agent libtool software-properties-common unzip wget zip

# Install clang
RUN \
  wget --quiet -O - http://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
  apt-add-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic main" && \
  apt-get install --no-install-recommends -y clang-6.0 libclang1-6.0 libllvm6.0 llvm-6.0 llvm-6.0-runtime

# Add build dependencies transferred from native Mac build server
RUN \
  mkdir -p /usr/local/sysroot-x86_64-apple-macosx10.13/usr && \
  cd /usr/local/sysroot-x86_64-apple-macosx10.13/usr && \
  wget --quiet -O - https://s3-eu-west-1.amazonaws.com/prelert-artifacts/dependencies/usr-x86_64-apple-macosx10.13-1.tar.bz2 | tar jxf - && \
  wget --quiet -O - https://s3-eu-west-1.amazonaws.com/prelert-artifacts/dependencies/xcode-x86_64-apple-macosx10.13-1.tar.bz2 | tar jxf -

# Build cctools-port
RUN \
  git clone https://github.com/tpoechtrager/cctools-port.git && \
  cd cctools-port/cctools && \
  git checkout 921-ld64-409.12 && \
  sed -i -e 's/autoconf/autoreconf -fi/' autogen.sh && \
  export CC=clang-6.0 && \
  export CXX=clang++-6.0 && \
  ./autogen.sh && \
  ./configure --target=x86_64-apple-macosx10.13 --with-llvm-config=/usr/bin/llvm-config-6.0 && \
  make -j`nproc` && \
  make install && \
  cd ../.. && \
  rm -rf cctools-port

