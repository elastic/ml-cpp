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

FROM docker.elastic.co/ml-dev/ml-linux-dependency-builder:4 AS builder

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

# Clone PyTorch and build LibTorch
# If the PyTorch branch is changed also update PYTORCH_BUILD_VERSION
ARG pytorch_branch=viable/strict

ENV PYTORCH_BRANCH=${pytorch_branch}
ENV PYTORCH_BUILD_VERSION=""

RUN \
  cd ${build_dir} && \
  git -c advice.detachedHead=false clone --depth=1 --branch=${PYTORCH_BRANCH} https://github.com/pytorch/pytorch.git && \
  cd pytorch && \
  git submodule sync && \
  git submodule update --init --recursive && \
  sed -i -e 's/system(/strlen(/' torch/csrc/jit/codegen/fuser/cpu/fused_kernel.cpp && \
  sed -i -e '104 i set(PYTHON_EXECUTABLE "/usr/local/bin/python3.12")' ./third_party/onnx/CMakeLists.txt && \
  if [[ "$PYTORCH_BRANCH" != "main" && "$PYTORCH_BRANCH" != "viable/strict" ]]; then PYTORCH_BUILD_VERSION=$(expr "$PYTORCH_BRANCH" : 'v\(.*\)'); fi && \
  export BLAS=MKL && \
  export BUILD_TEST=OFF && \
  export BUILD_CAFFE2=OFF && \
  export USE_NUMPY=OFF && \
  export USE_DISTRIBUTED=OFF && \
  export USE_MKLDNN=ON && \
  export USE_QNNPACK=OFF && \
  export USE_PYTORCH_QNNPACK=OFF && \
  export USE_XNNPACK=OFF && \
  export PYTORCH_BUILD_VERSION && \
  export PYTORCH_BUILD_NUMBER=1 && \
  export MAX_JOBS=10 && \
  /usr/local/bin/python3.12 setup.py install && \
  mkdir -p /usr/local/gcc133/include/pytorch && \
  /bin/cp -rf torch/include/* /usr/local/gcc133/include/pytorch/ && \
  /bin/cp -f torch/lib/libtorch_cpu.so /usr/local/gcc133/lib && \
  /bin/cp -f torch/lib/libc10.so /usr/local/gcc133/lib && \
  cd .. && \
  rm -rf pytorch

FROM rockylinux:8
COPY --from=builder /usr/local/gcc133 /usr/local/gcc133
RUN \
  dnf -y update && \
  dnf install -y bzip2 gcc git make unzip which zip zlib-devel findutils && \
  dnf clean all && \
  rm -rf /var/cache/dnf /tmp/*
