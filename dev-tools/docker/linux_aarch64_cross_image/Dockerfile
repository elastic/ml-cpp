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

FROM rockylinux:8

# This is basically automating the setup instructions in build-setup/linux.md

LABEL maintainer="Valeriy Khakhutskyy <valeriy.khakhutskyy@elastic.co>"

# Make sure OS packages are up to date and required packages are installed
RUN dnf -y update && \
    dnf install -y dnf-plugins-core && \
    dnf config-manager --set-enabled powertools && \
    dnf install -y bzip2 gcc gcc-c++ git make texinfo unzip wget which zip zlib-devel findutils bison flex file pkg-config autoconf automake libtool gettext && \
    dnf clean all && \
    rm -rf /var/cache/dnf

# Add build dependencies transferred from native aarch64 build server
RUN \
  mkdir -p /usr/local/sysroot-aarch64-linux-gnu/usr && \
  cd /usr/local/sysroot-aarch64-linux-gnu/usr && \
  wget --quiet -O - https://s3-eu-west-2.amazonaws.com/ml-cpp-artifacts/dependencies/usr-aarch64-linux-gnu-13.tar.bz2 | tar jxf - && \
  cd .. && \
  ln -s usr/lib lib && \
  ln -s usr/lib64 lib64

# For compiling with hardening and optimisation
ENV CFLAGS="-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse"
ENV CXXFLAGS="-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse"
ENV LDFLAGS="-Wl,-z,relro -Wl,-z,now"
ENV LDFLAGS_FOR_TARGET="-Wl,-z,relro -Wl,-z,now"

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
ENV PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin

# Build gcc 13.3 cross compiler
RUN \
  wget --quiet -O - http://ftpmirror.gnu.org/gcc/gcc-13.3.0/gcc-13.3.0.tar.gz | tar zxf - && \
  cd gcc-13.3.0 && \
  contrib/download_prerequisites && \
  sed -i -e 's/$(SHLIB_LDFLAGS)/-Wl,-z,relro -Wl,-z,now $(SHLIB_LDFLAGS)/' libgcc/config/t-slibgcc && \
  cd .. && \
  mkdir gcc-13.3.0-build && \
  cd gcc-13.3.0-build && \
  ../gcc-13.3.0/configure --prefix=/usr/local/gcc133 --with-sysroot=/usr/local/sysroot-aarch64-linux-gnu --target=aarch64-linux-gnu --enable-languages=c,c++ --enable-vtable-verify --with-system-zlib --disable-multilib && \
  make -j$(nproc) && \
  make install && \
  cd .. && \
  rm -rf gcc-13.3.0 gcc-13.3.0-build

# Build binutils
RUN \
  wget --quiet -O - http://ftpmirror.gnu.org/binutils/binutils-2.41.tar.bz2 | tar jxf - && \
  cd binutils-2.41 && \
  ./configure --prefix=/usr/local/gcc133 --with-sysroot=/usr/local/sysroot-aarch64-linux-gnu --target=aarch64-linux-gnu --enable-vtable-verify --with-system-zlib --disable-multilib --disable-libstdcxx --with-gcc-major-version-only && \
  make -j$(nproc) && \
  make install && \
  cd .. && \
  rm -rf binutils-2.41

# Build patchelf
RUN \
  wget --quiet -O - https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0.tar.bz2 | tar jxf - && \
  cd patchelf-0.18.0 && \
  ./configure --prefix=/usr/local/gcc133 && \
  make -j$(nproc) && \
  make install && \
  cd .. && \
  rm -rf patchelf-0.18.0

# Install CMake
RUN \
  wget --quiet https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-x86_64.sh && \
  chmod +x cmake-3.30.5-linux-x86_64.sh && \
  ./cmake-3.30.5-linux-x86_64.sh --skip-license --prefix=/usr/local/gcc133 && \
  rm -f cmake-3.30.5-linux-x86_64.sh
