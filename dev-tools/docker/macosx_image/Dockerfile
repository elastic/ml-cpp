#
# ELASTICSEARCH CONFIDENTIAL
#
# Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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
  apt-get install --no-install-recommends -y clang-3.8 libclang1-3.8 libllvm3.8 llvm-3.8 llvm-3.8-runtime

# Add build dependencies transferred from native Mac build server
RUN \
  mkdir -p /usr/local/sysroot-x86_64-apple-macosx10.10/usr && \
  cd /usr/local/sysroot-x86_64-apple-macosx10.10/usr && \
  wget --quiet -O - https://s3-eu-west-1.amazonaws.com/prelert-artifacts/dependencies/usr-x86_64-apple-macosx10.10-2.tar.bz2 | tar jxf - && \
  wget --quiet -O - https://s3-eu-west-1.amazonaws.com/prelert-artifacts/dependencies/xcode-x86_64-apple-macosx10.10-1.tar.bz2 | tar jxf -

# Build cctools-port
RUN \
  git clone https://github.com/tpoechtrager/cctools-port.git && \
  cd cctools-port/cctools && \
  git checkout 877.8-ld64-253.9 && \
  ./autogen.sh && \
  ./configure --target=x86_64-apple-macosx10.10 --with-llvm-config=/usr/bin/llvm-config-3.8 && \
  make -j`nproc` && \
  make install && \
  cd ../.. && \
  rm -rf cctools-port

