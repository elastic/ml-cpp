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
    dnf install -y bzip2 curl gcc gcc-c++ git libffi-devel make texinfo unzip wget which xz zip zlib-devel findutils

# Install sccache for GCS-backed compilation caching. Keep it under /usr/local/bin (not under
# /usr/local/gcc133) so it is not copied into the final image with the toolchain tree.
ARG SCCACHE_VERSION=v0.14.0
RUN mkdir -p /usr/local/bin && \
    curl -fsSL "https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
    | tar xz -C /usr/local/bin --strip-components=1 "sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl/sccache" && \
    chmod +x /usr/local/bin/sccache

# For compiling with hardening and optimisation
ENV CFLAGS="-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse"
ENV CXXFLAGS="-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse"
ENV LDFLAGS="-Wl,-z,relro -Wl,-z,now"
ENV LDFLAGS_FOR_TARGET="-Wl,-z,relro -Wl,-z,now"

ARG build_dir=/usr/src

# Update paths to use the compiler in C++17 mode. Keep /usr/local/bin on PATH in this stage so
# the sccache binary (installed above) is visible during the PyTorch RUN steps; sccache stays
# under /usr/local/bin (not under /usr/local/gcc133) so it is not copied into the final image with
# the toolchain tree.
ENV LD_LIBRARY_PATH=/usr/local/gcc133/lib64:/usr/local/gcc133/lib:/usr/lib:/lib
ENV PATH=/usr/local/gcc133/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
ENV CXX="g++ -std=gnu++17"

# Install Intel MKL into gcc133 (same pattern as dev-tools/docker/linux_image/Dockerfile) so
# libtorch_cpu.so resolves MKL at runtime in the final image after COPY --from=builder.
RUN \
  echo -e '[oneAPI]\n\
name=Intel oneAPI repository\n\
baseurl=https://yum.repos.intel.com/oneapi\n\
enabled=1\n\
gpgcheck=1\n\
repo_gpgcheck=1\n\
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB\n' > /etc/yum.repos.d/oneAPI.repo && \
  dnf install -y intel-oneapi-mkl-devel-2024.0 && \
  (cd /opt/intel/oneapi/mkl/2024.0 && tar cf - lib) | (cd /usr/local/gcc133 && tar xvf -) && \
  dnf clean all && \
  rm -rf /var/cache/dnf /opt/intel/oneapi/mkl/2024.0/doc /tmp/*

# Clone PyTorch and build LibTorch
# PYTORCH_BUILD_VERSION is only set for tagged branches (e.g. v2.7.1);
# for main/viable/strict PyTorch derives the version from version.txt.
ARG pytorch_branch=viable/strict
ARG pytorch_commit=unknown

ENV PYTORCH_BRANCH=${pytorch_branch}

# Split the build into clone + build so sccache (GCS-backed) can accelerate
# incremental rebuilds. The GCS key is passed via --mount=type=secret to
# avoid baking credentials into the image.
RUN \
  cd ${build_dir} && \
  git -c advice.detachedHead=false clone --depth=1 --branch=${PYTORCH_BRANCH} https://github.com/pytorch/pytorch.git && \
  cd pytorch && \
  git submodule sync && \
  git submodule update --init --recursive && \
  sed -i -e 's/system(/strlen(/' torch/csrc/jit/codegen/fuser/cpu/fused_kernel.cpp && \
  sed -i -e '104 i set(PYTHON_EXECUTABLE "/usr/local/bin/python3.12")' ./third_party/onnx/CMakeLists.txt

RUN --mount=type=secret,id=gcs_key \
  cd ${build_dir}/pytorch && \
  if [[ "$PYTORCH_BRANCH" != "main" && "$PYTORCH_BRANCH" != "viable/strict" ]]; then export PYTORCH_BUILD_VERSION=$(expr "$PYTORCH_BRANCH" : 'v\(.*\)'); fi && \
  export BLAS=MKL && \
  export BUILD_TEST=OFF && \
  export BUILD_CAFFE2=OFF && \
  export USE_NUMPY=OFF && \
  export USE_DISTRIBUTED=OFF && \
  export USE_MKLDNN=ON && \
  export USE_QNNPACK=OFF && \
  export USE_PYTORCH_QNNPACK=OFF && \
  export USE_XNNPACK=OFF && \
  export PYTORCH_BUILD_NUMBER=1 && \
  export MAX_JOBS=10 && \
  if [ -f /run/secrets/gcs_key ]; then \
    export SCCACHE_GCS_BUCKET=elastic-ml-cpp-sccache && \
    export SCCACHE_GCS_KEY_PREFIX=pytorch-build && \
    export SCCACHE_GCS_RW_MODE=READ_WRITE && \
    export SCCACHE_GCS_KEY_PATH=/run/secrets/gcs_key && \
    export CMAKE_C_COMPILER_LAUNCHER=sccache && \
    export CMAKE_CXX_COMPILER_LAUNCHER=sccache && \
    sccache --start-server && \
    echo "sccache: using GCS backend (bucket=$SCCACHE_GCS_BUCKET prefix=$SCCACHE_GCS_KEY_PREFIX)"; \
  else \
    echo "sccache: no GCS credentials — building without cache"; \
  fi && \
  /usr/local/bin/python3.12 -m pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org -r requirements.txt && \
  /usr/local/bin/python3.12 setup.py install && \
  if command -v sccache &>/dev/null && sccache --show-stats &>/dev/null; then sccache --show-stats; sccache --stop-server; fi && \
  mkdir -p /usr/local/gcc133/include/pytorch && \
  /bin/cp -rf torch/include/* /usr/local/gcc133/include/pytorch/ && \
  /bin/cp -f torch/lib/libtorch_cpu.so /usr/local/gcc133/lib && \
  /bin/cp -f torch/lib/libc10.so /usr/local/gcc133/lib && \
  cd .. && \
  rm -rf pytorch

FROM rockylinux:8
COPY --from=builder /usr/local/gcc133 /usr/local/gcc133
# Match linux_image final stage: MKL + libtorch under gcc133; needed for `import torch`
# when running without the full compile-time shell (e.g. allowlist validation).
ENV LD_LIBRARY_PATH=/usr/local/gcc133/lib64:/usr/local/gcc133/lib:/usr/lib:/lib
ENV PATH=/usr/local/gcc133/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
# Python 3.12 + torch site-packages for allowlist validation
COPY --from=builder /usr/local/bin/python3.12 /usr/local/bin/python3.12
COPY --from=builder /usr/local/bin/pip3.12 /usr/local/bin/pip3.12
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/lib/libpython3.12.a /usr/local/lib/libpython3.12.a
RUN ln -sf /usr/local/bin/python3.12 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip3

ARG pytorch_commit=unknown
LABEL pytorch.commit=${pytorch_commit}

RUN \
  dnf -y update && \
  dnf install -y bzip2 gcc git make openssl-libs unzip which zip zlib-devel findutils && \
  dnf clean all && \
  rm -rf /var/cache/dnf /tmp/*
