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
    dnf groupinstall "Development Tools" -y && \
    dnf install -y bzip2 gcc gcc-c++ git libffi-devel make texinfo unzip wget which zip zlib-devel && \
    dnf clean all && \
    rm -rf /var/cache/dnf /tmp/* 

# For compiling with hardening and optimisation
ENV CFLAGS="-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -march=armv8-a+crc+crypto"
ENV CXXFLAGS="-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -march=armv8-a+crc+crypto"
ENV LDFLAGS="-Wl,-z,relro -Wl,-z,now"
ENV LDFLAGS_FOR_TARGET="-Wl,-z,relro -Wl,-z,now"

ARG build_dir=/usr/src

# Build GCC 13.3
RUN \
  cd ${build_dir} && \
  wget --quiet -O - http://ftpmirror.gnu.org/gcc/gcc-13.3.0/gcc-13.3.0.tar.gz | tar zxf - && \
  cd gcc-13.3.0 && \
  contrib/download_prerequisites && \
  sed -i -e 's/$(SHLIB_LDFLAGS)/-Wl,-z,relro -Wl,-z,now $(SHLIB_LDFLAGS)/' libgcc/config/t-slibgcc && \
  cd .. && \
  mkdir gcc-13.3.0-build && \
  cd gcc-13.3.0-build && \
  ../gcc-13.3.0/configure --prefix=/usr/local/gcc133 --enable-languages=c,c++ --enable-vtable-verify --with-system-zlib --disable-lto-compression --without-system-liblzma --disable-multilib --without-xz --without-lzma --without-zstd && \
  make -j$(nproc) && \
  make install && \
  cd .. && \
  rm -rf gcc-13.3.0 gcc-13.3.0-build /tmp/*

# Update paths to use the newly built compiler in C++17 mode
ENV LD_LIBRARY_PATH=/usr/local/gcc133/lib64:/usr/local/gcc133/lib:/usr/lib:/lib
ENV PATH=/usr/local/gcc133/bin:/usr/bin:/bin:/usr/sbin:/sbin
ENV CXX="g++ -std=gnu++17"

# Build binutils
RUN \
  cd ${build_dir} && \
  wget --quiet -O - http://ftpmirror.gnu.org/binutils/binutils-2.41.tar.bz2 | tar jxf - && \
  cd binutils-2.41 && \
  ./configure --prefix=/usr/local/gcc133 --enable-vtable-verify --with-system-zlib --disable-libstdcxx --with-gcc-major-version-only --without-lzma --without-zstd && \
  make -j$(nproc) && \
  make install && \
  cd .. && \
  rm -rf binutils-2.41 /tmp/*

# Build libxml2
RUN \
  cd ${build_dir} && \
  wget --quiet -O - https://download.gnome.org/sources/libxml2/2.10/libxml2-2.10.4.tar.xz | tar Jxf - && \
  cd libxml2-2.10.4 && \
  ./configure --prefix=/usr/local/gcc133 --without-python --without-readline --without-lzma --without-zstd  && \
  make -j$(nproc) && \
  make install && \
  cd .. && \
  rm -rf libxml2-2.10.4 /tmp/*

RUN \
  for i in $(ls /lib64/libzstd.*); do mv $i "$i.bak" ; done && \
  for i in $(ls /lib64/liblzma.*); do mv $i "$i.bak" ; done

# Build Boost
RUN \
  cd ${build_dir} && \
  wget --quiet -O - https://boostorg.jfrog.io/artifactory/main/release/1.86.0/source/boost_1_86_0.tar.bz2 | tar jxf - && \
  cd boost_1_86_0 && \
  ./bootstrap.sh --without-libraries=context,coroutine,graph_parallel,mpi,python --without-icu && \
  sed -i -e 's/{13ul/{3ul, 13ul/' boost/unordered/detail/prime_fmod.hpp && \
  ./b2 -j$(nproc) --layout=versioned --disable-icu pch=off optimization=speed inlining=full \
    define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS \
    define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT \
    define=BOOST_LOG_WITHOUT_EVENT_LOG \
    define=BOOST_LOG_WITHOUT_SYSLOG \
    define=BOOST_LOG_WITHOUT_IPC \
    define=_FORTIFY_SOURCE=2 \
    cxxflags='-std=gnu++17 -fstack-protector -march=armv8-a+crc+crypto' \
    cflags='-D__STDC_FORMAT_MACROS' \
    linkflags='-std=gnu++17 -Wl,-z,relro -Wl,-z,now' && \
  ./b2 install --prefix=/usr/local/gcc133 --layout=versioned --disable-icu pch=off optimization=speed inlining=full \
    define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS \
    define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT \
    define=BOOST_LOG_WITHOUT_EVENT_LOG \
    define=BOOST_LOG_WITHOUT_SYSLOG \
    define=BOOST_LOG_WITHOUT_IPC \
    define=_FORTIFY_SOURCE=2 \
    cxxflags='-std=gnu++17 -fstack-protector -march=armv8-a+crc+crypto' \
    cflags='-D__STDC_FORMAT_MACROS' \
    linkflags='-std=gnu++17 -Wl,-z,relro -Wl,-z,now' && \
  cd .. && \
  rm -rf boost_1_86_0 /tmp/*

# Build patchelf
RUN \
  cd ${build_dir} && \
  wget --quiet -O - https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0.tar.bz2 | tar jxf - && \
  cd patchelf-0.18.0 && \
  ./configure --prefix=/usr/local/gcc133 && \
  make -j$(nproc) && \
  make install && \
  cd .. && \
  rm -rf patchelf-0.18.0 /tmp/*

# Build OpenSSL 1.1.1
# This is only needed as a dependency for Python 3.12 during the PyTorch build
# Not using --prefix=/usr/local/gcc133 so that this can be excluded from the final image
RUN \
  cd ${build_dir} && \
  wget --quiet -O - https://www.openssl.org/source/old/1.1.1/openssl-1.1.1u.tar.gz | tar xzf - && \
  cd openssl-1.1.1u && \
  ./Configure --prefix=/usr/local shared linux-aarch64 && \
  make -j$(nproc) && \
  make install && \
  cd .. && \
  rm -rf openssl-1.1.1u /usr/local/share/man /tmp/*

RUN \
  for i in $(ls /lib64/libzstd.*.bak); do mv $i "${i%.bak}" ; done && \
  for i in $(ls /lib64/liblzma.*.bak); do mv $i "${i%.bak}" ; done

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
  rm -rf Python-3.12.7 /usr/local/share/man /tmp/*

# Install Python dependencies
RUN \
  /usr/local/bin/pip3.12 install numpy ninja pyyaml setuptools cffi typing_extensions future six requests dataclasses && \
  rm -rf /root/.cache/pip /tmp/*

# Install CMake 3.30.5
RUN \
  cd ${build_dir} && \
  wget --quiet https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-aarch64.sh && \
  chmod +x cmake-3.30.5-linux-aarch64.sh && \
  ./cmake-3.30.5-linux-aarch64.sh --skip-license --prefix=/usr/local/gcc133 && \
  rm -f cmake-3.30.5-linux-aarch64.sh /tmp/*

# Clone PyTorch and build LibTorch
# If the PyTorch branch is changed also update PYTORCH_BUILD_VERSION
RUN \
  cd ${build_dir} && \
  git -c advice.detachedHead=false clone --depth=1 --branch=v2.5.1 https://github.com/pytorch/pytorch.git && \
  cd pytorch && \
  git submodule sync && \
  git submodule update --init --recursive && \
  sed -i -e 's/system(/strlen(/' torch/csrc/jit/codegen/fuser/cpu/fused_kernel.cpp && \
  sed -i -e '104 i set(PYTHON_EXECUTABLE "/usr/local/bin/python3.12")' ./third_party/onnx/CMakeLists.txt && \
  export BLAS=Eigen && \
  export BUILD_TEST=OFF && \
  export USE_FBGEMM=OFF && \
  export USE_KINETO=OFF && \
  export USE_DISTRIBUTED=OFF && \
  export USE_MKLDNN=ON && \
  export USE_QNNPACK=OFF && \
  export USE_PYTORCH_QNNPACK=OFF && \
  export PYTORCH_BUILD_VERSION=2.5.1 && \
  export PYTORCH_BUILD_NUMBER=1 && \
  export MAX_JOBS=10 && \
  /usr/local/bin/python3.12 setup.py install && \
  mkdir /usr/local/gcc133/include/pytorch && \
  cp -r torch/include/* /usr/local/gcc133/include/pytorch/ && \
  cp torch/lib/libtorch_cpu.so /usr/local/gcc133/lib && \
  cp torch/lib/libc10.so /usr/local/gcc133/lib && \
  cd .. && \
  rm -rf pytorch /tmp/*

FROM rockylinux:8
COPY --from=builder /usr/local/gcc133 /usr/local/gcc133
RUN \
  dnf -y update && \
  dnf install -y bzip2 gcc git make unzip which zip zlib-devel findutils && \
  dnf clean all && \
  rm -rf /var/cache/dnf /tmp/*
