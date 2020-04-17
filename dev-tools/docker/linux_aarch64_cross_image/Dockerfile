#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

FROM centos:7

# This is basically automating the setup instructions in build-setup/linux_aarch64_cross_compiled.md

MAINTAINER David Roberts <dave.roberts@elastic.co>

# Make sure OS packages are up to date and required packages are installed
RUN \
  rm /var/lib/rpm/__db.* && \
  yum install -y bzip2 gcc gcc-c++ git make texinfo unzip wget which zip zlib-devel

# Add build dependencies transferred from native aarch64 build server
RUN \
  mkdir -p /usr/local/sysroot-aarch64-linux-gnu/usr && \
  cd /usr/local/sysroot-aarch64-linux-gnu/usr && \
  wget --quiet -O - https://s3-eu-west-1.amazonaws.com/prelert-artifacts/dependencies/usr-aarch64-linux-gnu-1.tar.bz2 | tar jxf - && \
  cd .. && \
  ln -s usr/lib lib && \
  ln -s usr/lib64 lib64

# For compiling with hardening and optimisation
ENV CFLAGS "-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2"
ENV CXXFLAGS "-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2"
ENV LDFLAGS "-Wl,-z,relro -Wl,-z,now"
ENV LDFLAGS_FOR_TARGET "-Wl,-z,relro -Wl,-z,now"

# Build binutils (bootstrap version)
RUN \
  wget --quiet -O - http://ftpmirror.gnu.org/binutils/binutils-2.25.tar.bz2 | tar jxf - && \
  cd binutils-2.25 && \
  ./configure --with-sysroot=/usr/local/sysroot-aarch64-linux-gnu --target=aarch64-linux-gnu --with-system-zlib --disable-multilib --disable-libstdcxx && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf binutils-2.25

# Update the path so that the bootstrap cross tools are visible
ENV PATH /usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin

# Build gcc 7.5 cross compiler
RUN \
  wget --quiet -O - http://ftpmirror.gnu.org/gcc/gcc-7.5.0/gcc-7.5.0.tar.gz | tar zxf - && \
  cd gcc-7.5.0 && \
  contrib/download_prerequisites && \
  sed -i -e 's/$(SHLIB_LDFLAGS)/-Wl,-z,relro -Wl,-z,now $(SHLIB_LDFLAGS)/' libgcc/config/t-slibgcc && \
  cd .. && \
  mkdir gcc-7.5.0-build && \
  cd gcc-7.5.0-build && \
  ../gcc-7.5.0/configure --prefix=/usr/local/gcc75 --with-sysroot=/usr/local/sysroot-aarch64-linux-gnu --target=aarch64-linux-gnu --enable-languages=c,c++ --enable-vtable-verify --with-system-zlib --disable-multilib && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf gcc-7.5.0 gcc-7.5.0-build

# Build binutils (final version)
RUN \
  wget --quiet -O - http://ftpmirror.gnu.org/binutils/binutils-2.34.tar.bz2 | tar jxf - && \
  cd binutils-2.34 && \
  ./configure --prefix=/usr/local/gcc75 --with-sysroot=/usr/local/sysroot-aarch64-linux-gnu --target=aarch64-linux-gnu --enable-vtable-verify --with-system-zlib --disable-multilib --disable-libstdcxx --with-gcc-major-version-only && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf binutils-2.34

# Build patchelf
RUN \
  wget --quiet -O - http://nixos.org/releases/patchelf/patchelf-0.9/patchelf-0.9.tar.gz | tar zxf - && \
  cd patchelf-0.9 && \
  ./configure --prefix=/usr/local/gcc75 && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf patchelf-0.9

