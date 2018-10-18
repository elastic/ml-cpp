#!/usr/bin/env bash

# fail when any exit code != 0
set -e

NUMCPUS=`grep -c '^processor' /proc/cpuinfo`
MEM=$(free -g|awk '/^Mem:/{print $2}')
echo "Using $NUMCPUS cores and $MEM gb memory"

mkdir -p /home/vagrant/ml
mkdir -p /home/vagrant/ml/src

find /home/vagrant/ml -type d -exec chmod 755 {} \;
find /home/vagrant/ml -type d -exec chown vagrant:vagrant {} \;

# Install misc
if [ ! -f package.state ]; then
  echo "Installing misc packages..."
  add-apt-repository -y ppa:webupd8team/java
  apt-get update
  apt-get -qq -y install git wget build-essential maven unzip perl libbz2-1.0 libbz2-dev python-setuptools zlib1g-dev
  touch package.state
fi

# Install Java
if [ ! -f java.state ]; then
  echo "Installing Java..."
  echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections
  echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections
  apt-get -qq -y install oracle-java8-installer
  touch java.state
fi

# Install Gradle
if [ ! -f gradle.state ]; then
  echo "Installing Gradle..."
  echo "  Downloading Gradle..."
  wget --quiet -O gradle.zip https://services.gradle.org/distributions/gradle-3.3-all.zip
  unzip -qq gradle.zip -d gradle
  rm gradle.zip
  touch gradle.state
fi

# Various env variables
echo "Setting env variables..."
export CXX='g++ -std=gnu++0x'

# ----------------- Compile gcc 6.2 -------------------------
if [ ! -f gcc.state ]; then
  echo "Compiling GCC 6.2..."
  echo "  Downloading..."
  wget --quiet http://ftpmirror.gnu.org/gcc/gcc-6.2.0/gcc-6.2.0.tar.gz
  # gcc needs different source and build directories
  mkdir gcc-source
  tar xfz gcc-6.2.0.tar.gz -C gcc-source --strip-components=1
  cd gcc-source
  contrib/download_prerequisites
  cd ..
  mkdir gcc-build
  cd gcc-build
  echo "  Configuring..."
  ../gcc-source/configure --prefix=/usr/local/gcc62 --enable-languages=c,c++ --with-system-zlib --disable-multilib > configure.log 2>&1
  echo "  Making..."
  make -j$NUMCPUS --load-average=$NUMCPUS > make.log 2>&1
  make install > make_install.log 2>&1
  cd ..
  rm gcc-6.2.0.tar.gz
  touch gcc.state
fi

# Update paths to use the newly built compiler
export LD_LIBRARY_PATH=/usr/local/gcc62/lib64:/usr/local/gcc62/lib:/usr/lib:/lib
export PATH=/usr/local/gcc62/bin:/usr/bin:/bin:/usr/sbin:/sbin:/home/vagrant/bin

# ----------------- Compile libxml -------------------------
if [ ! -f libxml2.state ]; then
  echo "Compiling libxml..."
  echo "  Downloading..."
  wget --quiet -O LATEST_LIBXML2.tar.gz ftp://anonymous@ftp.xmlsoft.org/libxml2/LATEST_LIBXML2
  mkdir libxml
  tar xfz LATEST_LIBXML2.tar.gz -C libxml --strip-components=1
  cd libxml
  sed -i -e 's/-O2/-O3/g' configure
  echo "  Configuring..."
  ./configure --prefix=/usr/local/gcc62 --without-python --without-readline > configure.log 2>&1
  echo "  Making..."
  make -j$NUMCPUS --load-average=$NUMCPUS > make.log 2>&1
  make install > make_install.log 2>&1
  cd ..
  rm LATEST_LIBXML2.tar.gz
  touch libxml2.state
fi

# ----------------- Compile APR -------------------------
if [ ! -f apr.state ]; then
  echo "Compiling APR..."
  echo "  Downloading..."
  wget --quiet http://mirrors.sonic.net/apache//apr/apr-1.5.2.tar.gz
  mkdir apr
  tar xfz apr-1.5.2.tar.gz -C apr --strip-components=1
  cd apr
  echo "  Configuring..."
  ./configure --prefix=/usr/local/gcc62 > configure.log 2>&1
  echo "  Making..."
  make -j$NUMCPUS --load-average=$NUMCPUS > make.log 2>&1
  make install > make_install.log 2>&1
  cd ..
  rm apr-1.5.2.tar.gz
  touch apr.state
fi

# ----------------- Compile APR-Utilities -------------------------
if [ ! -f apr-util.state ]; then
  echo "Compiling APR-Utilities..."
  echo "  Downloading..."
  wget --quiet http://mirrors.sonic.net/apache//apr/apr-util-1.5.4.tar.gz
  mkdir apr-util
  tar xfz apr-util-1.5.4.tar.gz -C apr-util --strip-components=1
  cd apr-util
  echo "  Configuring..."
  ./configure --prefix=/usr/local/gcc62 --with-apr=/usr/local/gcc62/bin/apr-1-config --with-expat=builtin > configure.log 2>&1
  echo "  Making..."
  make -j$NUMCPUS --load-average=$NUMCPUS > make.log 2>&1
  make install > make_install.log 2>&1
  cd ..
  rm apr-util-1.5.4.tar.gz
  touch apr-util.state
fi


# ----------------- Compile log4cxx -------------------------
if [ ! -f log4cxx.state ]; then
  echo "Compiling log4cxx..."
  echo "  Downloading..."
  wget --quiet http://apache.cs.utah.edu/logging/log4cxx/0.10.0/apache-log4cxx-0.10.0.tar.gz
  mkdir log4cxx
  tar xfz apache-log4cxx-0.10.0.tar.gz -C log4cxx --strip-components=1

  cd log4cxx
  echo "  Configuring..."

  sed -i -e 's/#define _LOG4CXX_OBJECTPTR_INIT(x) { exchange(x);/#define _LOG4CXX_OBJECTPTR_INIT(x) : ObjectPtrBase() { exchange(x); /g' src/main/include/log4cxx/helpers/objectptr.h
  sed -i -e 's/#define _LOG4CXX_OBJECTPTR_INIT(x) : p(x) {/#define _LOG4CXX_OBJECTPTR_INIT(x) : ObjectPtrBase(), p(x) {/g' src/main/include/log4cxx/helpers/objectptr.h
  sed -i -e 's%#include <log4cxx/helpers/bytebuffer.h>%#include <log4cxx/helpers/bytebuffer.h>\n#include <string.h>%g' src/main/cpp/inputstreamreader.cpp
  sed -i -e 's%#include <log4cxx/helpers/bytebuffer.h>%#include <log4cxx/helpers/bytebuffer.h>\n#include <string.h>%g' src/main/cpp/socketoutputstream.cpp
  sed -i -e 's/#include <locale.h>/#include <locale.h>\n#include <string.h>\n#include <stdio.h>\n#include <wchar.h>/' src/examples/cpp/console.cpp

  sed -i -e '152,163s/0x/(char)0x/g' src/main/cpp/locationinfo.cpp
  sed -i -e '239,292s/0x/(char)0x/g' src/main/cpp/loggingevent.cpp
  sed -i -e '39s/0x/(char)0x/g' src/main/cpp/objectoutputstream.cpp
  sed -i -e '84,92s/0x/(char)0x/g' src/main/cpp/objectoutputstream.cpp
  sed -i -e '193,214s/0x/(char)0x/g' src/test/cpp/xml/domtestcase.cpp

  ./configure --prefix=/usr/local/gcc62 --with-charset=utf-8 --with-logchar=utf-8 --with-apr=/usr/local/gcc62 --with-apr-util=/usr/local/gcc62

  echo "  Making..."
  make -j$NUMCPUS --load-average=$NUMCPUS > make.log 2>&1
  make install > make_install.log 2>&1
  cd ..
  rm apache-log4cxx-0.10.0.tar.gz
  touch log4cxx.state
fi

# ----------------- Compile Boost -------------------------
if [ ! -f boost.state ]; then
  echo "Compiling Boost..."
  echo "  Downloading..."
  wget --quiet -O boost_1_65_1.tar.gz http://downloads.sourceforge.net/project/boost/boost/1.65.1/boost_1_65_1.tar.gz
  rm -rf boost; mkdir boost
  tar xfz boost_1_65_1.tar.gz -C boost --strip-components=1

  cd boost
  echo "  Bootstrapping..."
  ./bootstrap.sh cxxflags=-std=gnu++0x --without-libraries=context --without-libraries=coroutine --without-libraries=graph_parallel --without-libraries=log --without-libraries=mpi --without-libraries=python --without-icu > bootstrap.log 2>&1

  echo "  Configuring..."

  sed -i -e 's/(17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \\/(3ul)(17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \\/' boost/unordered/detail/util.hpp
  sed -i -e 's%#if ((defined(__linux__) \&\& !defined(__UCLIBC__) \&\& !defined(BOOST_MATH_HAVE_FIXED_GLIBC)) || defined(__QNX__) || defined(__IBMCPP__)) \&\& !defined(BOOST_NO_FENV_H)%#if ((!defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS) \&\& defined(__linux__) \&\& !defined(__UCLIBC__) \&\& !defined(BOOST_MATH_HAVE_FIXED_GLIBC)) || defined(__QNX__) || defined(__IBMCPP__)) \&\& !defined(BOOST_NO_FENV_H)%g' boost/math/tools/config.hpp

  echo "  Building..."
  ./b2 -j$NUMCPUS --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS > b2_make.log 2>&1
  ./b2 install --prefix=/usr/local/gcc62 --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS > b2_make_install.log 2>&1

  cd ..
  rm boost_1_65_1.tar.gz
  touch boost.state
fi


# ----------------- Compile cppunit -------------------------
if [ ! -f cppunit.state ]; then
  echo "Compiling cppunit..."
  echo "  Downloading..."
  wget --quiet http://dev-www.libreoffice.org/src/cppunit-1.13.2.tar.gz
  mkdir cppunit
  tar xfz cppunit-1.13.2.tar.gz -C cppunit --strip-components=1
  cd cppunit
  echo "  Configuring..."
  ./configure --prefix=/usr/local/gcc62 > configure.log 2>&1
  echo "  Making..."
  make -j$NUMCPUS --load-average=$NUMCPUS > make.log 2>&1
  make install > make_install.log 2>&1
  cd ..
  rm cppunit-1.13.2.tar.gz
  touch cppunit.state
fi

# ----------------- Compile patchelf -------------------------
if [ ! -f patchelf.state ]; then
  echo "Compiling patchelf..."
  echo "  Downloading..."
  wget --quiet http://nixos.org/releases/patchelf/patchelf-0.8/patchelf-0.8.tar.gz
  mkdir patchelf
  tar xfz patchelf-0.8.tar.gz -C patchelf --strip-components=1
  cd patchelf
  echo "  Configuring..."
  ./configure --prefix=/usr/local/gcc62 > configure.log 2>&1
  echo "  Making..."
  make -j$NUMCPUS --load-average=$NUMCPUS > make.log 2>&1
  make install > make_install.log 2>&1
  cd ..
  rm patchelf-0.8.tar.gz
  touch patchelf.state
fi

# ----------------- Compile gcovr -------------------------
if [ ! -f gcovr.state ]; then
  echo "Compiling gcovr..."
  echo "  Downloading..."
  wget --quiet https://pypi.python.org/packages/bc/47/3123c7a85a7df81501fbcfa13b49c240d6d4aa68ccd69851e985019e217d/gcovr-2.4.tar.gz#md5=672db629469882b93c40016aebff50ac
  mkdir gcovr
  tar xfz gcovr-2.4.tar.gz -C gcovr --strip-components=1
  cd gcovr
  python setup.py install
  cd ..
  rm gcovr-2.4.tar.gz
  touch gcovr.state
fi

# ------------------ bashrc -----------------------------
cat >> /home/vagrant/.bash_profile <<'EOF'
umask 0002
export ML_SRC_HOME=/home/vagrant/ml/src
export JAVA_HOME=/usr/lib/jvm/java-8-oracle
unset JAVA_ROOT
export GRADLE_HOME=/home/vagrant/gradle/gradle-3.3
export LD_LIBRARY_PATH=/usr/local/gcc62/lib64:/usr/local/gcc62/lib:/usr/lib:/lib
export PATH=$JAVA_HOME/bin:$GRADLE_HOME/bin:/usr/local/node-v6.9.0-linux-x64/bin:/usr/local/gcc62/bin:/usr/bin:/bin:/usr/sbin:/sbin:/home/vagrant/bin

ulimit -c unlimited
cd /home/vagrant/ml/src
EOF
