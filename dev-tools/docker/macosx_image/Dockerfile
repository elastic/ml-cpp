#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

FROM ubuntu:16.04

# This is basically automating the setup instructions in build-setup/macos_cross_compiled.md

MAINTAINER David Roberts <dave.roberts@elastic.co>

# Make sure apt-get is up to date and required packages are installed
RUN \
  apt-get update && \
  apt-get install --no-install-recommends -y apt-utils automake autogen build-essential bzip2 git gobjc libtool software-properties-common unzip wget zip

# Install clang
# Note: apt-key cannot update /etc/apt/trusted.gpg on the Ubuntu 16.04 Docker image, so it's updated in /tmp
RUN \
  cp /etc/apt/trusted.gpg /tmp && \
  wget --quiet -O - http://apt.llvm.org/llvm-snapshot.gpg.key | apt-key --keyring /tmp/trusted.gpg add - && \
  mv /tmp/trusted.gpg /etc/apt && \
  rm /tmp/trusted.gpg~ && \
  apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial main" && \
  apt-get install --no-install-recommends -y clang-3.9 libclang1-3.9 libllvm3.9 llvm-3.9 llvm-3.9-runtime

# Add build dependencies transferred from native Mac build server
RUN \
  mkdir -p /usr/local/sysroot-x86_64-apple-macosx10.11/usr && \
  cd /usr/local/sysroot-x86_64-apple-macosx10.11/usr && \
  wget --quiet -O - https://s3-eu-west-1.amazonaws.com/prelert-artifacts/dependencies/usr-x86_64-apple-macosx10.11-1.tar.bz2 | tar jxf - && \
  wget --quiet -O - https://s3-eu-west-1.amazonaws.com/prelert-artifacts/dependencies/xcode-x86_64-apple-macosx10.11-1.tar.bz2 | tar jxf -

# Build cctools-port
RUN \
  git clone https://github.com/tpoechtrager/cctools-port.git && \
  cd cctools-port/cctools && \
  git checkout 895-ld64-274.2 && \
  ./autogen.sh && \
  ./configure --target=x86_64-apple-macosx10.11 --with-llvm-config=/usr/bin/llvm-config-3.9 && \
  make -j`nproc` && \
  make install && \
  cd ../.. && \
  rm -rf cctools-port

