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

FROM centos:7 AS builder

# This is basically automating the setup instructions in build-setup/linux.md

MAINTAINER David Roberts <dave.roberts@elastic.co>

# Make sure OS packages are up to date and required packages are installed
# libffi and openssl are required for building Python
RUN \
  rm /var/lib/rpm/__db.* && \
  yum install -y bzip2 gcc gcc-c++ git libffi-devel make openssl-devel texinfo unzip wget which xz zip zlib-devel

# For compiling with hardening and optimisation
ENV CFLAGS "-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse"
ENV CXXFLAGS "-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse"
ENV LDFLAGS "-Wl,-z,relro -Wl,-z,now"
ENV LDFLAGS_FOR_TARGET "-Wl,-z,relro -Wl,-z,now"

ARG build_dir=/usr/src

# Build gcc 10.3
RUN \
  cd ${build_dir} && \
  wget --quiet -O - http://ftpmirror.gnu.org/gcc/gcc-10.3.0/gcc-10.3.0.tar.gz | tar zxf - && \
  cd gcc-10.3.0 && \
  contrib/download_prerequisites && \
  sed -i -e 's/$(SHLIB_LDFLAGS)/-Wl,-z,relro -Wl,-z,now $(SHLIB_LDFLAGS)/' libgcc/config/t-slibgcc && \
  cd .. && \
  mkdir gcc-10.3.0-build && \
  cd gcc-10.3.0-build && \
  ../gcc-10.3.0/configure --prefix=/usr/local/gcc103 --enable-languages=c,c++ --enable-vtable-verify --with-system-zlib --disable-multilib && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf gcc-10.3.0 gcc-10.3.0-build

# Update paths to use the newly built compiler in C++17 mode
ENV LD_LIBRARY_PATH /usr/local/gcc103/lib64:/usr/local/gcc103/lib:/usr/lib:/lib
ENV PATH /usr/local/gcc103/bin:/usr/bin:/bin:/usr/sbin:/sbin
ENV CXX "g++ -std=gnu++17"

# Build binutils
RUN \
  cd ${build_dir} && \
  wget --quiet -O - http://ftpmirror.gnu.org/binutils/binutils-2.37.tar.bz2 | tar jxf - && \
  cd binutils-2.37 && \
  ./configure --prefix=/usr/local/gcc103 --enable-vtable-verify --with-system-zlib --disable-libstdcxx --with-gcc-major-version-only && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf binutils-2.37

# Build libxml2
RUN \
  cd ${build_dir} && \
  wget --quiet --no-check-certificate -O - https://download.gnome.org/sources/libxml2/2.9/libxml2-2.9.14.tar.xz | tar Jxf - && \
  cd libxml2-2.9.14 && \
  ./configure --prefix=/usr/local/gcc103 --without-python --without-readline && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf libxml2-2.9.14

# Build Boost
RUN \
  cd ${build_dir} && \
  wget --quiet -O - https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.bz2 | tar jxf - && \
  cd boost_1_77_0 && \
  ./bootstrap.sh --without-libraries=context --without-libraries=coroutine --without-libraries=graph_parallel --without-libraries=mpi --without-libraries=python --without-icu && \
  sed -i -e 's/(17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \\/(3ul)(17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \\/' boost/unordered/detail/implementation.hpp && \
  ./b2 -j`nproc` --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC define=_FORTIFY_SOURCE=2 cxxflags='-std=gnu++17 -fstack-protector -msse4.2 -mfpmath=sse' linkflags='-std=gnu++17 -Wl,-z,relro -Wl,-z,now' && \
  ./b2 install --prefix=/usr/local/gcc103 --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC define=_FORTIFY_SOURCE=2 cxxflags='-std=gnu++17 -fstack-protector -msse4.2 -mfpmath=sse' linkflags='-std=gnu++17 -Wl,-z,relro -Wl,-z,now' && \
  cd .. && \
  rm -rf boost_1_77_0

# Build patchelf
RUN \
  cd ${build_dir} && \
  wget --quiet -O - https://github.com/NixOS/patchelf/releases/download/0.13/patchelf-0.13.tar.bz2 | tar jxf - && \
  cd patchelf-0.13.20210805.a949ff2 && \
  ./configure --prefix=/usr/local/gcc103 && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf patchelf-0.13.20210805.a949ff2

# Build Python 3.7
# --enable-optimizations for a stable/release build
# Not using --prefix=/usr/local/gcc103 so that this can be excluded from the final image
RUN \
  cd ${build_dir} && \
  wget --quiet -O - https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz | tar xzf - && \
  cd Python-3.7.9 && \
  ./configure --enable-optimizations && \
  make altinstall && \
  cd .. && \
  rm -rf Python-3.7.9

# Install Python dependencies
RUN \
  /usr/local/bin/pip3.7 install numpy ninja pyyaml setuptools cffi typing_extensions future six requests dataclasses

# Install CMake
# v3.19.2 minimum is required
RUN \
  cd ${build_dir} && \
  wget --quiet https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2-Linux-x86_64.sh && \
  chmod +x cmake-3.23.2-Linux-x86_64.sh && \
  ./cmake-3.23.2-Linux-x86_64.sh --skip-license --prefix=/usr/local/gcc103 && \
  rm -f cmake-3.23.2-Linux-x86_64.sh

# Install Intel MKL
RUN \
  yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo && \
  yum -y install intel-mkl-2020.4-912 && \
  cp /opt/intel/mkl/lib/intel64/libmkl*.so /usr/local/gcc103/lib

# Clone PyTorch and build LibTorch
# If the PyTorch branch is changed also update PYTORCH_BUILD_VERSION
RUN \
  cd ${build_dir} && \
  git -c advice.detachedHead=false clone --depth=1 --branch=v1.11.0 https://github.com/pytorch/pytorch.git && \
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
  export USE_BREAKPAD=OFF && \
  export PYTORCH_BUILD_VERSION=1.11.0 && \
  export PYTORCH_BUILD_NUMBER=1 && \
  /usr/local/bin/python3.7 setup.py install && \
  mkdir /usr/local/gcc103/include/pytorch && \
  cp -r torch/include/* /usr/local/gcc103/include/pytorch/ && \
  cp torch/lib/libtorch_cpu.so /usr/local/gcc103/lib && \
  cp torch/lib/libc10.so /usr/local/gcc103/lib && \
  cd .. && \
  rm -rf pytorch

FROM centos:7
COPY --from=builder /usr/local/gcc103 /usr/local/gcc103
RUN \
  rm /var/lib/rpm/__db.* && \
  yum install -y bzip2 gcc git make unzip which zip zlib-devel
