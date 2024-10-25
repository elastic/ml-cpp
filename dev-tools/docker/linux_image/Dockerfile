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

FROM rockylinux:8 AS builder

# This is basically automating the setup instructions in build-setup/linux.md

LABEL maintainer="Valeriy Khakhutskyy <valeriy.khakhutskyy@elastic.co>"

# Make sure OS packages are up to date and required packages are installed
# libffi is required for building Python
RUN dnf -y update && \
    dnf install -y dnf-plugins-core && \
    dnf config-manager --set-enabled powertools && \
    dnf install -y bzip2 gcc gcc-c++ git libffi-devel make texinfo unzip wget which xz zip zlib-devel gcc-toolset-12

# Enable gcc-toolset-12
ENV PATH=/opt/rh/gcc-toolset-12/root/usr/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-12/root/usr/lib64:$LD_LIBRARY_PATH

# For compiling with hardening and optimisation
ENV CFLAGS "-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse"
ENV CXXFLAGS "-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse"
ENV LDFLAGS "-Wl,-z,relro -Wl,-z,now"
ENV LDFLAGS_FOR_TARGET "-Wl,-z,relro -Wl,-z,now"

ARG build_dir=/usr/src

# Build GCC 12.4
RUN \
  cd ${build_dir} && \
  wget --quiet -O - http://ftpmirror.gnu.org/gcc/gcc-12.4.0/gcc-12.4.0.tar.gz | tar zxf - && \
  cd gcc-12.4.0 && \
  contrib/download_prerequisites && \
  sed -i -e 's/$(SHLIB_LDFLAGS)/-Wl,-z,relro -Wl,-z,now $(SHLIB_LDFLAGS)/' libgcc/config/t-slibgcc && \
  cd .. && \
  mkdir gcc-12.4.0-build && \
  cd gcc-12.4.0-build && \
  ../gcc-12.4.0/configure --prefix=/usr/local/gcc124 --enable-languages=c,c++ --enable-vtable-verify --with-system-zlib --disable-multilib && \
  make -j$(nproc) && \
  make install && \
  cd .. && \
  rm -rf gcc-12.4.0 gcc-12.4.0-build

# Update paths to use the newly built compiler in C++17 mode
ENV LD_LIBRARY_PATH=/usr/local/gcc124/lib64:/usr/local/gcc124/lib:/usr/lib:/lib
ENV PATH=/usr/local/gcc124/bin:/usr/bin:/bin:/usr/sbin:/sbin
ENV CXX="g++ -std=gnu++17"

# Build binutils
RUN \
  cd ${build_dir} && \
  wget --quiet -O - http://ftpmirror.gnu.org/binutils/binutils-2.41.tar.bz2 | tar jxf - && \
  cd binutils-2.41 && \
  ./configure --prefix=/usr/local/gcc124 --enable-vtable-verify --with-system-zlib --disable-libstdcxx --with-gcc-major-version-only && \
  make -j$(nproc) && \
  make install && \
  cd .. && \
  rm -rf binutils-2.41

# Build libxml2
RUN \
  cd ${build_dir} && \
  wget --quiet -O - https://download.gnome.org/sources/libxml2/2.10/libxml2-2.10.4.tar.xz | tar Jxf - && \
  cd libxml2-2.10.4 && \
  ./configure --prefix=/usr/local/gcc124 --without-python --without-readline && \
  make -j$(nproc) && \
  make install && \
  cd .. && \
  rm -rf libxml2-2.10.4

# Build Boost
RUN \
  cd ${build_dir} && \
  wget --quiet -O - https://boostorg.jfrog.io/artifactory/main/release/1.83.0/source/boost_1_83_0.tar.bz2 | tar jxf - && \
  cd boost_1_83_0 && \
  ./bootstrap.sh --without-libraries=context,coroutine,graph_parallel,mpi,python --without-icu && \
  sed -i -e 's|(13ul)(29ul)(53ul)(97ul)(193ul)(389ul)(769ul)(1543ul)(3079ul)(6151ul)(|\(3ul)(13ul)(29ul)(53ul)(97ul)(193ul)(389ul)(769ul)(1543ul)(3079ul)(6151ul)(|' boost/unordered/detail/prime_fmod.hpp && \
  ./b2 -j$(nproc) --layout=versioned --disable-icu pch=off optimization=speed inlining=full \
    define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS \
    define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT \
    define=BOOST_LOG_WITHOUT_EVENT_LOG \
    define=BOOST_LOG_WITHOUT_SYSLOG \
    define=BOOST_LOG_WITHOUT_IPC \
    define=_FORTIFY_SOURCE=2 \
    cxxflags='-std=gnu++17 -fstack-protector -msse4.2 -mfpmath=sse' \
    cflags='-D__STDC_FORMAT_MACROS' \
    linkflags='-std=gnu++17 -Wl,-z,relro -Wl,-z,now' && \
  ./b2 install --prefix=/usr/local/gcc124 --layout=versioned --disable-icu pch=off optimization=speed inlining=full \
    define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS \
    define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT \
    define=BOOST_LOG_WITHOUT_EVENT_LOG \
    define=BOOST_LOG_WITHOUT_SYSLOG \
    define=BOOST_LOG_WITHOUT_IPC \
    define=_FORTIFY_SOURCE=2 \
    cxxflags='-std=gnu++17 -fstack-protector -msse4.2 -mfpmath=sse' \
    cflags='-D__STDC_FORMAT_MACROS' \
    linkflags='-std=gnu++17 -Wl,-z,relro -Wl,-z,now' && \
  cd .. && \
  rm -rf boost_1_83_0

# Build patchelf
RUN \
  cd ${build_dir} && \
  wget --quiet -O - https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0.tar.bz2 | tar jxf - && \
  cd patchelf-0.18.0 && \
  ./configure --prefix=/usr/local/gcc124 && \
  make -j$(nproc) && \
  make install && \
  cd .. && \
  rm -rf patchelf-0.18.0

# Build OpenSSL 1.1.1
# This is only needed as a dependency for Python 3.10 during the PyTorch build
# Not using --prefix=/usr/local/gcc124 so that this can be excluded from the final image
RUN \
  cd ${build_dir} && \
  wget --quiet -O - https://www.openssl.org/source/old/1.1.1/openssl-1.1.1u.tar.gz | tar xzf - && \
  cd openssl-1.1.1u && \
  ./Configure --prefix=/usr/local shared linux-x86_64 && \
  make -j$(nproc) && \
  make install && \
  cd .. && \
  rm -rf openssl-1.1.1u

# Build Python 3.10
# --enable-optimizations for a stable/release build
# Not using --prefix=/usr/local/gcc124 so that this can be excluded from the final image
RUN \
  cd ${build_dir} && \
  wget --quiet -O - https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz | tar xzf - && \
  cd Python-3.10.13 && \
  sed -i -e 's~ssldir/lib~ssldir/lib64~' configure && \
  ./configure --enable-optimizations --with-openssl=/usr/local --with-openssl-rpath=/usr/local/lib64 && \
  make -j$(nproc) && \
  make altinstall && \
  cd .. && \
  rm -rf Python-3.10.13

# Install Python dependencies
RUN \
  /usr/local/bin/pip3.10 install numpy ninja pyyaml setuptools cffi typing_extensions future six requests dataclasses

# Install CMake 3.27.9
RUN \
  cd ${build_dir} && \
  wget --quiet https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.sh && \
  chmod +x cmake-3.27.9-linux-x86_64.sh && \
  ./cmake-3.27.9-linux-x86_64.sh --skip-license --prefix=/usr/local/gcc124 && \
  rm -f cmake-3.27.9-linux-x86_64.sh

# Install Intel MKL
RUN \
  echo -e '[oneAPI]\n\
name=Intel oneAPI repository\n\
baseurl=https://yum.repos.intel.com/oneapi\n\
enabled=1\n\
gpgcheck=1\n\
repo_gpgcheck=1\n\
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB\n' > /etc/yum.repos.d/oneAPI.repo && \
  yum install -y intel-oneapi-mkl-devel-2024.0 && \
  (cd /opt/intel/oneapi/mkl/2024.0 && tar cf - lib) | (cd /usr/local/gcc124 && tar xvf -)

# Clone PyTorch and build LibTorch
# If the PyTorch branch is changed also update PYTORCH_BUILD_VERSION
RUN \
  cd ${build_dir} && \
  git -c advice.detachedHead=false clone --depth=1 --branch=v2.3.1 https://github.com/pytorch/pytorch.git && \
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
  export PYTORCH_BUILD_VERSION=2.3.1 && \
  export PYTORCH_BUILD_NUMBER=1 && \
  export MAX_JOBS=10 && \
  /usr/local/bin/python3.10 setup.py install && \
  mkdir /usr/local/gcc124/include/pytorch && \
  cp -r torch/include/* /usr/local/gcc124/include/pytorch/ && \
  cp torch/lib/libtorch_cpu.so /usr/local/gcc124/lib && \
  cp torch/lib/libc10.so /usr/local/gcc124/lib && \
  cd .. && \
  rm -rf pytorch

FROM rockylinux:8
COPY --from=builder /usr/local/gcc124 /usr/local/gcc124
RUN yum -y update && \
    yum install -y bzip2 gcc git make unzip which zip zlib-devel
