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

FROM alpine:3.8

# Docker image containing the correct clang-format 5.0.1 for check-style.sh on aarch64

MAINTAINER Valeriy Khakhutskyy <valeriy.khakhutskyy@elastic.co>

# Install build dependencies
RUN apk update && \
    apk add --no-cache \
        bash \
        git \
        build-base \
        cmake \
        ninja \
        python3 \
        libc-dev \
        linux-headers \
        wget \
        tar \
        xz

# Build clang-format 5.0.1 from LLVM source
# We need to build from source since Alpine 3.8 doesn't have clang-format 5.0.1 for aarch64
ARG LLVM_VERSION=5.0.1
ARG BUILD_DIR=/tmp/llvm-build

RUN mkdir -p ${BUILD_DIR} && \
    cd ${BUILD_DIR} && \
    # Download LLVM 5.0.1 source from official releases (using the old repository structure before monorepo)
    # LLVM and Clang were separate repositories for version 5.0.1
    wget -q https://releases.llvm.org/5.0.1/llvm-5.0.1.src.tar.xz && \
    tar -xJf llvm-5.0.1.src.tar.xz && \
    mv llvm-5.0.1.src llvm && \
    cd llvm/tools && \
    wget -q https://releases.llvm.org/5.0.1/cfe-5.0.1.src.tar.xz && \
    tar -xJf cfe-5.0.1.src.tar.xz && \
    mv cfe-5.0.1.src clang && \
    cd ../.. && \
    # Configure and build only clang-format (minimal build)
    mkdir build && \
    cd build && \
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_TARGETS_TO_BUILD="AArch64" \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        ../llvm && \
    # Build only clang-format to reduce build time and image size
    ninja clang-format && \
    # Install clang-format
    install -m 755 bin/clang-format /usr/local/bin/clang-format && \
    # Clean up build artifacts
    cd / && \
    rm -rf ${BUILD_DIR} && \
    # Install runtime libraries needed by clang-format before removing build-base
    # libstdc++ and libgcc_s are required for C++ runtime
    apk add --no-cache libstdc++ libgcc && \
    # Remove build-only dependencies (keep runtime libraries installed above)
    apk del build-base cmake ninja python3 wget tar xz

# Verify clang-format version
RUN clang-format --version | grep -q "5.0.1" || (echo "ERROR: clang-format version mismatch" && exit 1)

WORKDIR /ml-cpp
ENV CPP_SRC_HOME=/ml-cpp

