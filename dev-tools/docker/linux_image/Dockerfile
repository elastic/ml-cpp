#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

FROM centos:6

# This is basically automating the setup instructions in build-setup/linux.md

MAINTAINER David Roberts <dave.roberts@elastic.co>

# Make sure OS packages are up to date and required packages are installed
RUN \
  rm /var/lib/rpm/__db.* && \
  yum install -y gcc gcc-c++ git unzip wget zip zlib-devel

# For compiling with hardening and optimisation
ENV CFLAGS "-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2"
ENV CXXFLAGS "-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2"
ENV LDFLAGS "-Wl,-z,relro -Wl,-z,now"
ENV LDFLAGS_FOR_TARGET "-Wl,-z,relro -Wl,-z,now"

# Build gcc 7.3
RUN \
  wget --quiet -O - http://ftpmirror.gnu.org/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz | tar zxf - && \
  cd gcc-7.3.0 && \
  contrib/download_prerequisites && \
  sed -i -e 's/$(SHLIB_LDFLAGS)/$(LDFLAGS) $(SHLIB_LDFLAGS)/' libgcc/config/t-slibgcc && \
  cd .. && \
  mkdir gcc-7.3.0-build && \
  cd gcc-7.3.0-build && \
  ../gcc-7.3.0/configure --prefix=/usr/local/gcc73 --enable-languages=c,c++ --enable-vtable-verify --with-system-zlib --disable-multilib && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf gcc-7.3.0 gcc-7.3.0-build

# Update paths to use the newly built compiler in C++14 mode
ENV LD_LIBRARY_PATH /usr/local/gcc73/lib64:/usr/local/gcc73/lib:/usr/lib:/lib
ENV PATH /usr/local/gcc73/bin:/usr/bin:/bin:/usr/sbin:/sbin
ENV CXX "g++ -std=gnu++14"

# Build libxml2
RUN \
  wget --quiet -O - ftp://anonymous@ftp.xmlsoft.org/libxml2/libxml2-2.9.4.tar.gz | tar zxf - && \
  cd libxml2-2.9.4 && \
  ./configure --prefix=/usr/local/gcc73 --without-python --without-readline && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf libxml2-2.9.4

# Build expat
RUN \
  wget --quiet -O - http://github.com/libexpat/libexpat/releases/download/R_2_2_6/expat-2.2.6.tar.bz2 | tar jxf - && \
  cd expat-2.2.6 && \
  ./configure --prefix=/usr/local/gcc73 && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf expat-2.2.6

# Build APR
RUN \
  wget --quiet -O - http://archive.apache.org/dist/apr/apr-1.7.0.tar.bz2 | tar jxf - && \
  cd apr-1.7.0 && \
  sed -i -e "s/for ac_lib in '' crypt ufc; do/for ac_lib in ''; do/" configure && \
  sed -i -e 's/#define APR_HAVE_CRYPT_H         @crypth@/#define APR_HAVE_CRYPT_H         0/' include/apr.h.in && \
  ./configure --prefix=/usr/local/gcc73 && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf apr-1.7.0

# Build APR utilities
RUN \
  wget --quiet -O - http://archive.apache.org/dist/apr/apr-util-1.6.1.tar.bz2 | tar jxf - && \
  cd apr-util-1.6.1 && \
  sed -i -e "s/for ac_lib in '' crypt ufc; do/for ac_lib in ''; do/" configure && \
  sed -i -e 's/#define CRYPT_MISSING 0/#define CRYPT_MISSING 1/' crypto/apr_passwd.c && \
  ./configure --prefix=/usr/local/gcc73 --with-apr=/usr/local/gcc73/bin/apr-1-config --with-expat=/usr/local/gcc73 && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf apr-util-1.6.1

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
  ./configure --prefix=/usr/local/gcc73 --with-charset=utf-8 --with-logchar=utf-8 --with-apr=/usr/local/gcc73 --with-apr-util=/usr/local/gcc73 && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf apache-log4cxx-0.10.0

# Build Boost
RUN \
  wget --quiet -O - http://sourceforge.mirrorservice.org/b/bo/boost/boost/1.65.1/boost_1_65_1.tar.bz2 | tar jxf - && \
  cd boost_1_65_1 && \
  ./bootstrap.sh --without-libraries=context --without-libraries=coroutine --without-libraries=graph_parallel --without-libraries=log --without-libraries=mpi --without-libraries=python --without-icu && \
  sed -i -e 's/(17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \\/(3ul)(17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \\/' boost/unordered/detail/implementation.hpp && \
  sed -i -e 's%#if ((defined(__linux__) \&\& !defined(__UCLIBC__) \&\& !defined(BOOST_MATH_HAVE_FIXED_GLIBC)) || defined(__QNX__) || defined(__IBMCPP__)) \&\& !defined(BOOST_NO_FENV_H)%#if ((!defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS) \&\& defined(__linux__) \&\& !defined(__UCLIBC__) \&\& !defined(BOOST_MATH_HAVE_FIXED_GLIBC)) || defined(__QNX__) || defined(__IBMCPP__)) \&\& !defined(BOOST_NO_FENV_H)%g' boost/math/tools/config.hpp && \
  ./b2 -j`grep -c '^processor' /proc/cpuinfo` --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=_FORTIFY_SOURCE=2 cxxflags=-std=gnu++14 cxxflags=-fstack-protector linkflags=-Wl,-z,relro linkflags=-Wl,-z,now && \
  ./b2 install --prefix=/usr/local/gcc73 --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=_FORTIFY_SOURCE=2 cxxflags=-std=gnu++14 cxxflags=-fstack-protector linkflags=-Wl,-z,relro linkflags=-Wl,-z,now && \
  cd .. && \
  rm -rf boost_1_65_1

# Build cppunit
RUN \
  wget --quiet -O - http://dev-www.libreoffice.org/src/cppunit-1.13.2.tar.gz | tar zxf - && \
  cd cppunit-1.13.2 && \
  ./configure --prefix=/usr/local/gcc73 && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf cppunit-1.13.2

# Build patchelf
RUN \
  wget --quiet -O - http://nixos.org/releases/patchelf/patchelf-0.9/patchelf-0.9.tar.gz | tar zxf - && \
  cd patchelf-0.9 && \
  ./configure --prefix=/usr/local/gcc73 && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf patchelf-0.9

