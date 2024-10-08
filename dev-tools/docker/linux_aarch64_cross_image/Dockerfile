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

FROM centos:7

# This is basically automating the setup instructions in build-setup/linux_aarch64_cross_compiled.md

MAINTAINER David Roberts <dave.roberts@elastic.co>

# Make sure OS packages are up to date and required packages are installed
RUN \
  sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-* && \
  sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-* && \
  yum repolist enabled && \
  rm /var/lib/rpm/__db.* && \
  yum install -y bzip2 gcc gcc-c++ git make texinfo unzip wget which zip zlib-devel

# Add build dependencies transferred from native aarch64 build server
RUN \
  mkdir -p /usr/local/sysroot-aarch64-linux-gnu/usr && \
  cd /usr/local/sysroot-aarch64-linux-gnu/usr && \
  wget --quiet -O - https://s3-eu-west-2.amazonaws.com/ml-cpp-artifacts/dependencies/usr-aarch64-linux-gnu-13.tar.bz2 | tar jxf - && \
  cd .. && \
  ln -s usr/lib lib && \
  ln -s usr/lib64 lib64

# For compiling with hardening and optimisation
ENV CFLAGS "-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse"
ENV CXXFLAGS "-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse"
ENV LDFLAGS "-Wl,-z,relro -Wl,-z,now"
ENV LDFLAGS_FOR_TARGET "-Wl,-z,relro -Wl,-z,now"

# Build binutils (bootstrap version)
RUN \
  wget --quiet -O - http://ftpmirror.gnu.org/binutils/binutils-2.27.tar.bz2 | tar jxf - && \
  cd binutils-2.27 && \
  ./configure --with-sysroot=/usr/local/sysroot-aarch64-linux-gnu --target=aarch64-linux-gnu --with-system-zlib --disable-multilib --disable-libstdcxx && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf binutils-2.27

# Update the path so that the bootstrap cross tools are visible
ENV PATH /usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin

# Build gcc 10.3 cross compiler
RUN \
  wget --quiet -O - http://ftpmirror.gnu.org/gcc/gcc-10.3.0/gcc-10.3.0.tar.gz | tar zxf - && \
  cd gcc-10.3.0 && \
  contrib/download_prerequisites && \
  sed -i -e 's/$(SHLIB_LDFLAGS)/-Wl,-z,relro -Wl,-z,now $(SHLIB_LDFLAGS)/' libgcc/config/t-slibgcc && \
  cd .. && \
  mkdir gcc-10.3.0-build && \
  cd gcc-10.3.0-build && \
  ../gcc-10.3.0/configure --prefix=/usr/local/gcc103 --with-sysroot=/usr/local/sysroot-aarch64-linux-gnu --target=aarch64-linux-gnu --enable-languages=c,c++ --enable-vtable-verify --with-system-zlib --disable-multilib && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf gcc-10.3.0 gcc-10.3.0-build

# Build binutils (final version)
RUN \
  wget --quiet -O - http://ftpmirror.gnu.org/binutils/binutils-2.37.tar.bz2 | tar jxf - && \
  cd binutils-2.37 && \
  ./configure --prefix=/usr/local/gcc103 --with-sysroot=/usr/local/sysroot-aarch64-linux-gnu --target=aarch64-linux-gnu --enable-vtable-verify --with-system-zlib --disable-multilib --disable-libstdcxx --with-gcc-major-version-only && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf binutils-2.37

# Build patchelf
RUN \
  wget --quiet -O - https://github.com/NixOS/patchelf/releases/download/0.13/patchelf-0.13.tar.bz2 | tar jxf - && \
  cd patchelf-0.13.20210805.a949ff2 && \
  ./configure --prefix=/usr/local/gcc103 && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf patchelf-0.13.20210805.a949ff2

# Install CMake
# v3.19.2 minimum is required
RUN \
  cd ${build_dir} && \
  wget --quiet https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2-Linux-x86_64.sh && \
  chmod +x cmake-3.23.2-Linux-x86_64.sh && \
  ./cmake-3.23.2-Linux-x86_64.sh --skip-license --prefix=/usr/local/gcc103 && \
  rm -f cmake-3.23.2-Linux-x86_64.sh

