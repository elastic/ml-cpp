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

FROM ubuntu:22.04 AS builder

# This is basically automating the setup instructions in build-setup/linux.md

LABEL maintainer="Valeriy Khakhutskyy <valeriy.khakhutskyy@elastic.co>"

# Make sure OS packages are up to date and required packages are installed
# libffi is required for building Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bzip2 gcc g++ git libffi-dev make texinfo unzip wget xz-utils zip zlib1g-dev ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# For compiling with hardening and optimisation
ENV CFLAGS="-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse"
ENV CXXFLAGS="-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse"
ENV LDFLAGS="-Wl,-z,relro -Wl,-z,now"
ENV LDFLAGS_FOR_TARGET="-Wl,-z,relro -Wl,-z,now"

ARG build_dir=/usr/src

# Build gcc 13.3
RUN \
  cd ${build_dir} && \
  wget --quiet -O - https://ftp.mpi-inf.mpg.de/mirrors/gnu/mirror/gcc.gnu.org/pub/gcc/releases/gcc-13.3.0/gcc-13.3.0.tar.gz | tar xzf - && \
  cd gcc-13.3.0 && \
  contrib/download_prerequisites && \
  sed -i -e 's/$(SHLIB_LDFLAGS)/-Wl,-z,relro -Wl,-z,now $(SHLIB_LDFLAGS)/' libgcc/config/t-slibgcc && \
  cd .. && \
  mkdir gcc-13.3.0-build && \
  cd gcc-13.3.0-build && \
  ../gcc-13.3.0/configure --prefix=/usr/local/gcc133 --enable-languages=c,c++ --enable-vtable-verify --with-system-zlib --disable-multilib && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf gcc-13.3.0 gcc-13.3.0-build

# Update paths to use the newly built compiler in C++17 mode
ENV LD_LIBRARY_PATH=/usr/local/gcc133/lib64:/usr/local/gcc133/lib:/usr/lib:/lib
ENV PATH=/usr/local/gcc133/bin:/usr/bin:/bin:/usr/sbin:/sbin
ENV CXX="g++ -std=gnu++17"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates file bison flex && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Build binutils
RUN \
  cd ${build_dir} && \
  wget --quiet -O - http://ftpmirror.gnu.org/binutils/binutils-2.41.tar.bz2 | tar jxf - && \
  cd binutils-2.41 && \
  ./configure --prefix=/usr/local/gcc133 --enable-vtable-verify --with-system-zlib --disable-libstdcxx --with-gcc-major-version-only && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf binutils-2.41

# Build libxml2
RUN \
  cd ${build_dir} && \
  wget --quiet --no-check-certificate -O - https://download.gnome.org/sources/libxml2/2.10/libxml2-2.10.4.tar.xz | tar Jxf - && \
  cd libxml2-2.10.4 && \
  ./configure --prefix=/usr/local/gcc133 --without-python --without-readline && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf libxml2-2.10.4

# Build Boost
RUN \
  cd ${build_dir} && \
  wget --quiet -O - https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.bz2 | tar jxf - && \
  cd boost_1_86_0 && \
  ./bootstrap.sh --without-libraries=context --without-libraries=coroutine --without-libraries=graph_parallel --without-libraries=mpi --without-libraries=python --without-icu && \
  sed -i -e 's|(13ul)(29ul)(53ul)(97ul)(193ul)(389ul)(769ul)(1543ul)(3079ul)(6151ul)( \\|(3ul)(13ul)(29ul)(53ul)(97ul)(193ul)(389ul)(769ul)(1543ul)(3079ul)(6151ul)(       \\|' boost/unordered/detail/prime_fmod.hpp && \
  ./b2 -j`nproc` --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC define=_FORTIFY_SOURCE=2 cxxflags='-std=gnu++17 -fstack-protector -msse4.2 -mfpmath=sse' cflags='-D__STDC_FORMAT_MACROS' linkflags='-std=gnu++17 -Wl,-z,relro -Wl,-z,now' && \
  ./b2 install --prefix=/usr/local/gcc133 --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC define=_FORTIFY_SOURCE=2 cxxflags='-std=gnu++17 -fstack-protector -msse4.2 -mfpmath=sse' cflags='-D__STDC_FORMAT_MACROS' linkflags='-std=gnu++17 -Wl,-z,relro -Wl,-z,now' && \
  cd .. && \
  rm -rf boost_1_86_0

# Build patchelf
RUN \
  cd ${build_dir} && \
  wget --quiet -O - https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0.tar.bz2 | tar jxf - && \
  cd patchelf-0.18.0 && \
  ./configure --prefix=/usr/local/gcc133 && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf patchelf-0.18.0

# Build OpenSSL 1.1.1
# This is only needed as a dependency for Python 3.12 during the PyTorch build
# Not using --prefix=/usr/local/gcc133 so that this can be excluded from the final image
RUN \
  cd ${build_dir} && \
  wget --quiet --no-check-certificate -O - https://www.openssl.org/source/old/1.1.1/openssl-1.1.1u.tar.gz | tar xzf - && \
  cd openssl-1.1.1u && \
  ./Configure --prefix=/usr/local shared linux-x86_64 && \
  make -j`nproc` && \
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
  ./configure --enable-optimizations --with-openssl=/usr/local --with-openssl-rpath=/usr/local/lib && \
  make -j`nproc` && \
  make altinstall && \
  cd .. && \
  rm -rf Python-3.12.7

# Install Python dependencies
RUN \
  /usr/local/bin/pip3.12 install numpy ninja pyyaml setuptools cffi typing_extensions future six requests dataclasses

# Install CMake
# v3.30.5 minimum is required
RUN \
  cd ${build_dir} && \
  wget --quiet --no-check-certificate https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-x86_64.sh && \
  chmod +x cmake-3.30.5-linux-x86_64.sh && \
  ./cmake-3.30.5-linux-x86_64.sh --skip-license --prefix=/usr/local/gcc133 && \
  rm -f cmake-3.30.5-linux-x86_64.sh

ENV LD_LIBRARY_PATH=/usr/local/gcc133/lib:/usr/lib:/lib
ENV PATH=/usr/bin:/bin:/usr/sbin:/sbin

RUN apt-get update && \ 
    apt-get install -y gpg-agent && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Intel MKL
RUN \
  wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
  echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
  apt-get update && \
  apt-get install -y intel-oneapi-mkl-devel-2024.0 && \
  (cd /opt/intel/oneapi/mkl/2024.0 && tar cf - lib) | (cd /usr/local/gcc133 && tar xvf -)

ENV LD_LIBRARY_PATH=/usr/local/gcc133/lib64:/usr/local/gcc133/lib:/usr/lib:/lib
ENV PATH=/usr/local/gcc133/bin:/usr/bin:/bin:/usr/sbin:/sbin

# Clone PyTorch and build LibTorch
# If the PyTorch branch is changed also update PYTORCH_BUILD_VERSION
RUN \
  cd ${build_dir} && \
  git -c advice.detachedHead=false clone --depth=1 --branch=v2.5.1 https://github.com/pytorch/pytorch.git && \
  cd pytorch && \
  git submodule sync && \
  git submodule update --init --recursive && \
  sed -i -e 's/system(/strlen(/' torch/csrc/jit/codegen/fuser/cpu/fused_kernel.cpp && \
  export BLAS=MKL && \
  export BUILD_TEST=OFF && \
  export BUILD_CAFFE2=OFF && \
  export USE_NUMPY=OFF && \
  export USE_DISTRIBUTED=OFF && \
  export USE_MKLDNN=ON && \
  export USE_QNNPACK=OFF && \
  export USE_PYTORCH_QNNPACK=OFF && \
  export USE_XNNPACK=OFF && \
  export PYTORCH_BUILD_VERSION=2.5.1 && \
  export PYTORCH_BUILD_NUMBER=1 && \
  export MAX_JOBS=10 && \
  /usr/local/bin/python3.12 setup.py install
RUN cd ${build_dir}/pytorch && \
  mkdir /usr/local/gcc133/include/pytorch && \
  cp -r torch/include/* /usr/local/gcc133/include/pytorch/ && \
  cp torch/lib/libtorch_cpu.so /usr/local/gcc133/lib && \
  cp torch/lib/libc10.so /usr/local/gcc133/lib && \
  cd .. && \
  rm -rf pytorch

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

FROM ubuntu:22.04
COPY --from=builder /usr/local/gcc133 /usr/local/gcc133

ENV PATH=/usr/local/gcc133/bin:$PATH

RUN \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  bzip2 gcc git make unzip zip zlib1g-dev cmake ccache \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
