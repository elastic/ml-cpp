#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

FROM centos:7

# This is basically automating the setup instructions in build-setup/linux.md

MAINTAINER David Roberts <dave.roberts@elastic.co>

# Make sure OS packages are up to date and required packages are installed
RUN \
  rm /var/lib/rpm/__db.* && \
  yum install -y bzip2 gcc gcc-c++ git make texinfo unzip wget which zip zlib-devel

# For compiling with hardening and optimisation
ENV CFLAGS "-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2"
ENV CXXFLAGS "-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2"
ENV LDFLAGS "-Wl,-z,relro -Wl,-z,now"
ENV LDFLAGS_FOR_TARGET "-Wl,-z,relro -Wl,-z,now"

# Build gcc 7.5
RUN \
  wget --quiet -O - http://ftpmirror.gnu.org/gcc/gcc-7.5.0/gcc-7.5.0.tar.gz | tar zxf - && \
  cd gcc-7.5.0 && \
  contrib/download_prerequisites && \
  sed -i -e 's/$(SHLIB_LDFLAGS)/-Wl,-z,relro -Wl,-z,now $(SHLIB_LDFLAGS)/' libgcc/config/t-slibgcc && \
  cd .. && \
  mkdir gcc-7.5.0-build && \
  cd gcc-7.5.0-build && \
  ../gcc-7.5.0/configure --prefix=/usr/local/gcc75 --enable-languages=c,c++ --enable-vtable-verify --with-system-zlib --disable-multilib && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf gcc-7.5.0 gcc-7.5.0-build

# Update paths to use the newly built compiler in C++14 mode
ENV LD_LIBRARY_PATH /usr/local/gcc75/lib64:/usr/local/gcc75/lib:/usr/lib:/lib
ENV PATH /usr/local/gcc75/bin:/usr/bin:/bin:/usr/sbin:/sbin
ENV CXX "g++ -std=gnu++14"

# Build binutils
RUN \
  wget --quiet -O - http://ftpmirror.gnu.org/binutils/binutils-2.34.tar.bz2 | tar jxf - && \
  cd binutils-2.34 && \
  ./configure --prefix=/usr/local/gcc75 --enable-vtable-verify --with-system-zlib --disable-libstdcxx --with-gcc-major-version-only && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf binutils-2.34

# Build libxml2
RUN \
  wget --quiet -O - ftp://anonymous@ftp.xmlsoft.org/libxml2/libxml2-2.9.4.tar.gz | tar zxf - && \
  cd libxml2-2.9.4 && \
  ./configure --prefix=/usr/local/gcc75 --without-python --without-readline && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf libxml2-2.9.4

# Build Boost
RUN \
  wget --quiet -O - http://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.bz2 | tar jxf - && \
  cd boost_1_71_0 && \
  ./bootstrap.sh --without-libraries=context --without-libraries=coroutine --without-libraries=graph_parallel --without-libraries=mpi --without-libraries=python --without-icu && \
  sed -i -e 's/(17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \\/(3ul)(17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \\/' boost/unordered/detail/implementation.hpp && \
  ./b2 -j`nproc` --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC define=_FORTIFY_SOURCE=2 cxxflags=-std=gnu++14 cxxflags=-fstack-protector linkflags=-Wl,-z,relro linkflags=-Wl,-z,now && \
  ./b2 install --prefix=/usr/local/gcc75 --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC define=_FORTIFY_SOURCE=2 cxxflags=-std=gnu++14 cxxflags=-fstack-protector linkflags=-Wl,-z,relro linkflags=-Wl,-z,now && \
  cd .. && \
  rm -rf boost_1_71_0

# Build patchelf
RUN \
  wget --quiet -O - http://nixos.org/releases/patchelf/patchelf-0.9/patchelf-0.9.tar.gz | tar zxf - && \
  cd patchelf-0.9 && \
  ./configure --prefix=/usr/local/gcc75 && \
  make -j`nproc` && \
  make install && \
  cd .. && \
  rm -rf patchelf-0.9

