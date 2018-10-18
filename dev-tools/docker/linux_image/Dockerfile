#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

FROM droberts195/fedora:10

# This is basically automating the setup instructions in build-setup/linux.md

MAINTAINER David Roberts <dave.roberts@elastic.co>

# Make sure OS packages are up to date and required packages are installed
RUN \
  rm /var/lib/rpm/__db.* && \
  yum install -y coreutils curl diffutils file gcc gcc-c++ git libselinux rpm sed tar unzip which zip zlib-devel

# For compiling in C++11 mode
ENV CXX "g++ -std=gnu++0x"

# Build gcc 6.2
RUN \
  wget --quiet -O - http://ftpmirror.gnu.org/gcc/gcc-6.2.0/gcc-6.2.0.tar.gz | tar zxf - && \
  cd gcc-6.2.0 && \
  contrib/download_prerequisites && \
  cd .. && \
  mkdir gcc-6.2.0-build && \
  cd gcc-6.2.0-build && \
  ../gcc-6.2.0/configure --prefix=/usr/local/gcc62 --enable-languages=c,c++ --with-system-zlib --disable-multilib && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf gcc-6.2.0 gcc-6.2.0-build

# Update paths to use the newly built compiler
ENV LD_LIBRARY_PATH /usr/local/gcc62/lib64:/usr/local/gcc62/lib:/usr/lib:/lib
ENV PATH /usr/local/gcc62/bin:/usr/bin:/bin:/usr/sbin:/sbin

# Build libxml2
RUN \
  wget --quiet -O - ftp://anonymous@ftp.xmlsoft.org/libxml2/libxml2-2.9.4.tar.gz | tar zxf - && \
  cd libxml2-2.9.4 && \
  ./configure --prefix=/usr/local/gcc62 --without-python --without-readline && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf libxml2-2.9.4

# Build APR
RUN \
  wget --quiet -O - http://archive.apache.org/dist/apr/apr-1.5.2.tar.bz2 | tar jxf - && \
  cd apr-1.5.2 && \
  ./configure --prefix=/usr/local/gcc62 && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf apr-1.5.2

# Build APR utilities
RUN \
  wget --quiet -O - http://archive.apache.org/dist/apr/apr-util-1.5.4.tar.bz2 | tar jxf - && \
  cd apr-util-1.5.4 && \
  ./configure --prefix=/usr/local/gcc62 --with-apr=/usr/local/gcc62/bin/apr-1-config --with-expat=builtin && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf apr-util-1.5.4

# Build log4cxx
RUN \
  wget --quiet -O - http://apache.cs.utah.edu/logging/log4cxx/0.10.0/apache-log4cxx-0.10.0.tar.gz  | tar zxf - && \
  cd apache-log4cxx-0.10.0 && \
  sed -i -e 's/#define _LOG4CXX_OBJECTPTR_INIT(x) { exchange(x);/#define _LOG4CXX_OBJECTPTR_INIT(x) : ObjectPtrBase() { exchange(x); /g' src/main/include/log4cxx/helpers/objectptr.h && \
  sed -i -e 's/#define _LOG4CXX_OBJECTPTR_INIT(x) : p(x) {/#define _LOG4CXX_OBJECTPTR_INIT(x) : ObjectPtrBase(), p(x) {/g' src/main/include/log4cxx/helpers/objectptr.h && \
  sed -i -e 's%#include <log4cxx/helpers/bytebuffer.h>%#include <log4cxx/helpers/bytebuffer.h>\n#include <string.h>%g' src/main/cpp/inputstreamreader.cpp && \
  sed -i -e 's%#include <log4cxx/helpers/bytebuffer.h>%#include <log4cxx/helpers/bytebuffer.h>\n#include <string.h>%g' src/main/cpp/socketoutputstream.cpp && \
  sed -i -e 's/#include <locale.h>/#include <locale.h>\n#include <string.h>\n#include <stdio.h>\n#include <wchar.h>/' src/examples/cpp/console.cpp && \
  sed -i -e '152,163s/0x/(char)0x/g' src/main/cpp/locationinfo.cpp && \
  sed -i -e '239,292s/0x/(char)0x/g' src/main/cpp/loggingevent.cpp && \
  sed -i -e '39s/0x/(char)0x/g' src/main/cpp/objectoutputstream.cpp && \
  sed -i -e '84,92s/0x/(char)0x/g' src/main/cpp/objectoutputstream.cpp && \
  sed -i -e '193,214s/0x/(char)0x/g' src/test/cpp/xml/domtestcase.cpp && \
  ./configure --prefix=/usr/local/gcc62 --with-charset=utf-8 --with-logchar=utf-8 --with-apr=/usr/local/gcc62 --with-apr-util=/usr/local/gcc62 && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf apache-log4cxx-0.10.0

# Build Boost
RUN \
  wget --quiet -O - http://sourceforge.mirrorservice.org/b/bo/boost/boost/1.65.1/boost_1_65_1.tar.bz2 | tar jxf - && \
  cd boost_1_65_1 && \
  ./bootstrap.sh cxxflags=-std=gnu++0x --without-libraries=context --without-libraries=coroutine --without-libraries=graph_parallel --without-libraries=log --without-libraries=mpi --without-libraries=python --without-icu && \
  sed -i -e 's/(17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \\/(3ul)(17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \\/' boost/unordered/detail/implementation.hpp && \
  sed -i -e 's%#if ((defined(__linux__) \&\& !defined(__UCLIBC__) \&\& !defined(BOOST_MATH_HAVE_FIXED_GLIBC)) || defined(__QNX__) || defined(__IBMCPP__)) \&\& !defined(BOOST_NO_FENV_H)%#if ((!defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS) \&\& defined(__linux__) \&\& !defined(__UCLIBC__) \&\& !defined(BOOST_MATH_HAVE_FIXED_GLIBC)) || defined(__QNX__) || defined(__IBMCPP__)) \&\& !defined(BOOST_NO_FENV_H)%g' boost/math/tools/config.hpp && \
  ./b2 -j`grep -c '^processor' /proc/cpuinfo` --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS && \
  ./b2 install --prefix=/usr/local/gcc62 --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS && \
  cd .. && \
  rm -rf boost_1_65_1

# Build cppunit
RUN \
  wget --no-check-certificate --quiet -O - http://dev-www.libreoffice.org/src/cppunit-1.13.2.tar.gz | tar zxf - && \
  cd cppunit-1.13.2 && \
  ./configure --prefix=/usr/local/gcc62 && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf cppunit-1.13.2

# Build patchelf
RUN \
  wget --no-check-certificate --quiet -O - http://nixos.org/releases/patchelf/patchelf-0.8/patchelf-0.8.tar.gz | tar zxf - && \
  cd patchelf-0.8 && \
  ./configure --prefix=/usr/local/gcc62 && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf patchelf-0.8

