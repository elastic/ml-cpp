#!/usr/bin/env bash

# fail when any exit code != 0
set -e

NUMCPUS=`nproc`
MEM=$(free -g | awk '/^Mem:/{print $2}')
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
  apt-get -qq -y install git wget build-essential maven unzip perl libbz2-1.0 libbz2-dev bzip2 python-setuptools texinfo zlib1g-dev
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

# Env variables for hardening and optimisation
echo "Setting env variables..."
export CFLAGS='-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse'
export CXXFLAGS='-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse'
export LDFLAGS='-Wl,-z,relro -Wl,-z,now'
export LDFLAGS_FOR_TARGET='-Wl,-z,relro -Wl,-z,now'
unset LIBRARY_PATH

# ----------------- Compile gcc 9.3 -------------------------
if [ ! -f gcc.state ]; then
  echo "Compiling GCC 9.3..."
  echo "  Downloading..."
  wget --quiet http://ftpmirror.gnu.org/gcc/gcc-9.3.0/gcc-9.3.0.tar.gz
  # gcc needs different source and build directories
  mkdir gcc-source
  tar xfz gcc-9.3.0.tar.gz -C gcc-source --strip-components=1
  cd gcc-source
  contrib/download_prerequisites
  sed -i -e 's/$(SHLIB_LDFLAGS)/-Wl,-z,relro -Wl,-z,now $(SHLIB_LDFLAGS)/' libgcc/config/t-slibgcc
  cd ..
  mkdir gcc-build
  cd gcc-build
  echo "  Configuring..."
  ../gcc-source/configure --prefix=/usr/local/gcc93 --enable-languages=c,c++ --enable-vtable-verify --with-system-zlib --disable-multilib > configure.log 2>&1
  echo "  Making..."
  make -j$NUMCPUS --load-average=$NUMCPUS > make.log 2>&1
  make install > make_install.log 2>&1
  cd ..
  rm gcc-9.3.0.tar.gz
  touch gcc.state
fi

# Update paths to use the newly built compiler in C++17 mode
export LD_LIBRARY_PATH=/usr/local/gcc93/lib64:/usr/local/gcc93/lib:/usr/lib:/lib
export PATH=/usr/local/gcc93/bin:/usr/bin:/bin:/usr/sbin:/sbin:/home/vagrant/bin
export CXX='g++ -std=gnu++17'

# ----------------- Compile binutils -------------------------
if [ ! -f binutils.state ]; then
  echo "Compiling binutils..."
  echo "  Downloading..."
  wget --quiet -O binutils-2.34.tar.gz http://ftpmirror.gnu.org/binutils/binutils-2.34.tar.gz
  mkdir binutils
  tar xfz binutils-2.34.tar.gz -C binutils --strip-components=1
  cd binutils
  echo "  Configuring..."
  ./configure --prefix=/usr/local/gcc93 --enable-vtable-verify --with-system-zlib --disable-libstdcxx --with-gcc-major-version-only > configure.log 2>&1
  echo "  Making..."
  make -j$NUMCPUS --load-average=$NUMCPUS > make.log 2>&1
  make install > make_install.log 2>&1
  cd ..
  rm binutils-2.34.tar.gz
  touch binutils.state
fi

# ----------------- Compile libxml2 -------------------------
if [ ! -f libxml2.state ]; then
  echo "Compiling libxml..."
  echo "  Downloading..."
  wget --quiet -O libxml2-2.9.7.tar.gz ftp://anonymous@ftp.xmlsoft.org/libxml2/libxml2-2.9.7
  mkdir libxml
  tar xfz libxml2-2.9.7.tar.gz -C libxml --strip-components=1
  cd libxml
  echo "  Configuring..."
  ./configure --prefix=/usr/local/gcc93 --without-python --without-readline > configure.log 2>&1
  echo "  Making..."
  make -j$NUMCPUS --load-average=$NUMCPUS > make.log 2>&1
  make install > make_install.log 2>&1
  cd ..
  rm libxml2-2.9.7.tar.gz
  touch libxml2.state
fi

# ----------------- Compile Boost -------------------------
if [ ! -f boost.state ]; then
  echo "Compiling Boost..."
  echo "  Downloading..."
  wget --quiet -O boost_1_71_0.tar.gz https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.gz
  rm -rf boost; mkdir boost
  tar xfz boost_1_71_0.tar.gz -C boost --strip-components=1

  cd boost
  echo "  Bootstrapping..."
  ./bootstrap.sh --without-libraries=context --without-libraries=coroutine --without-libraries=graph_parallel --without-libraries=mpi --without-libraries=python --without-icu > bootstrap.log 2>&1

  echo "  Configuring..."

  sed -i -e 's/(17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \\/(3ul)(17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \\/' boost/unordered/detail/util.hpp
  sed -i -e 's%#if ((defined(__linux__) \&\& !defined(__UCLIBC__) \&\& !defined(BOOST_MATH_HAVE_FIXED_GLIBC)) || defined(__QNX__) || defined(__IBMCPP__)) \&\& !defined(BOOST_NO_FENV_H)%#if ((!defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS) \&\& defined(__linux__) \&\& !defined(__UCLIBC__) \&\& !defined(BOOST_MATH_HAVE_FIXED_GLIBC)) || defined(__QNX__) || defined(__IBMCPP__)) \&\& !defined(BOOST_NO_FENV_H)%g' boost/math/tools/config.hpp

  echo "  Building..."
  ./b2 -j$NUMCPUS --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC define=_FORTIFY_SOURCE=2 cxxflags='-std=gnu++17 -fstack-protector -msse4.2 -mfpmath=sse' linkflags='-std=gnu++17 -Wl,-z,relro -Wl,-z,now' > b2_make.log 2>&1
  ./b2 install --prefix=/usr/local/gcc93 --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC define=_FORTIFY_SOURCE=2 cxxflags='-std=gnu++17 -fstack-protector -msse4.2 -mfpmath=sse' linkflags='-std=gnu++17 -Wl,-z,relro -Wl,-z,now' > b2_make_install.log 2>&1

  cd ..
  rm boost_1_71_0.tar.gz
  touch boost.state
fi

# ----------------- Compile patchelf -------------------------
if [ ! -f patchelf.state ]; then
  echo "Compiling patchelf..."
  echo "  Downloading..."
  wget --quiet http://nixos.org/releases/patchelf/patchelf-0.10/patchelf-0.10.tar.gz
  mkdir patchelf
  tar xfz patchelf-0.9.tar.gz -C patchelf --strip-components=1
  cd patchelf
  echo "  Configuring..."
  ./configure --prefix=/usr/local/gcc93 > configure.log 2>&1
  echo "  Making..."
  make -j$NUMCPUS --load-average=$NUMCPUS > make.log 2>&1
  make install > make_install.log 2>&1
  cd ..
  rm patchelf-0.10.tar.gz
  touch patchelf.state
fi

# ------------------ bashrc -----------------------------
cat >> /home/vagrant/.bash_profile <<'EOF'
umask 0002
export ML_SRC_HOME=/home/vagrant/ml/src
export JAVA_HOME=/usr/lib/jvm/java-8-oracle
unset JAVA_ROOT
export LD_LIBRARY_PATH=/usr/local/gcc93/lib64:/usr/local/gcc93/lib:/usr/lib:/lib
export PATH=$JAVA_HOME/bin:/bin:/usr/local/gcc93/bin:/usr/bin:/bin:/usr/sbin:/sbin:/home/vagrant/bin

ulimit -c unlimited
cd /home/vagrant/ml/src
EOF
