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

# Increment the version here when a new tools/3rd party components image is built
FROM docker.elastic.co/ml-dev/ml-linux-build:31 AS builder

# This is basically automating the setup instructions in build-setup/linux.md

LABEL maintainer="Ed Savage <ed.savage@elastic.co>"

# Make sure OS packages are up to date and required packages are installed
# libffi is required for building Python
RUN dnf -y update && \
    dnf install -y dnf-plugins-core && \
    dnf config-manager --set-enabled powertools && \
    dnf install -y bzip2 gcc gcc-c++ git libffi-devel make texinfo unzip wget which xz zip zlib-devel findutils

# For compiling with hardening and optimisation
ENV CFLAGS="-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse"
ENV CXXFLAGS="-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse"
ENV LDFLAGS="-Wl,-z,relro -Wl,-z,now"
ENV LDFLAGS_FOR_TARGET="-Wl,-z,relro -Wl,-z,now"

ARG build_dir=/usr/src

# Update paths to use the compiler in C++17 mode
ENV LD_LIBRARY_PATH=/usr/local/gcc133/lib64:/usr/local/gcc133/lib:/usr/lib:/lib
ENV PATH=/usr/local/gcc133/bin:/usr/bin:/bin:/usr/sbin:/sbin
ENV CXX="g++ -std=gnu++17"

# Build OpenSSL 1.1.1
# This is only needed as a dependency for Python 3.12 during the PyTorch build
# Not using --prefix=/usr/local/gcc133 so that this can be excluded from the final image
RUN \
  cd ${build_dir} && \
  wget --quiet --no-check-certificate -O - https://www.openssl.org/source/old/1.1.1/openssl-1.1.1u.tar.gz | tar xzf - && \
  cd openssl-1.1.1u && \
  ./Configure --prefix=/usr/local shared linux-x86_64 && \
  make -j$(nproc) && \
  make install && \
  cd .. && \
  rm -rf openssl-1.1.1u

# Build Python 3.12
# --enable-optimizations for a stable/release build
# Not using --prefix=/usr/local/gcc133 so that this can be excluded from the final image
RUN \
  cd ${build_dir} && \
  wget --quiet -O - https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tgz | tar xzf - && \
  cd Python-3.12.7 && \
  sed -i -e 's~ssldir/lib~ssldir/lib64~' configure && \
  ./configure --enable-optimizations --with-openssl=/usr/local --with-openssl-rpath=/usr/local/lib64 && \
  make -j$(nproc) && \
  make altinstall && \
  cd .. && \
  rm -rf Python-3.12.7

# Install Python dependencies
RUN \
  /usr/local/bin/pip3.12 install numpy ninja pyyaml setuptools cffi typing_extensions future six requests dataclasses

# Install Intel MKL
RUN \
  echo -e '[oneAPI]\n\
name=Intel oneAPI repository\n\
baseurl=https://yum.repos.intel.com/oneapi\n\
enabled=1\n\
gpgcheck=1\n\
repo_gpgcheck=1\n\
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB\n' > /etc/yum.repos.d/oneAPI.repo && \
  dnf install -y intel-oneapi-mkl-devel-2024.0 && \
  (cd /opt/intel/oneapi/mkl/2024.0 && tar cf - lib) | (cd /usr/local/gcc133 && tar xvf -)
