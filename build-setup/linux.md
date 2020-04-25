# Machine Learning Build Machine Setup for Linux

To ensure everything is consistent for redistributable builds we build all redistributable components from source with a specific version of gcc.

You will need the following environment variables to be defined:

- `JAVA_HOME` - Should point to the JDK you want to use to run Gradle.
- `CPP_SRC_HOME` - Only required if building the C++ code directly using `make`, as Gradle sets it automatically.
- `PATH` - Must have `/usr/local/gcc62/bin` before `/usr/bin` and `/bin`.
- `LD_LIBRARY_PATH` - Must have `/usr/local/gcc62/lib64` and `/usr/local/gcc62/lib` before `/usr/lib` and `/lib`.

For example, you might create a `.bashrc` file in your home directory containing something like this:

```
umask 0002
export JAVA_HOME=/usr/local/jdk1.8.0_121
export LD_LIBRARY_PATH=/usr/local/gcc62/lib64:/usr/local/gcc62/lib:/usr/lib:/lib
export PATH=$JAVA_HOME/bin:/usr/local/gcc62/bin:/usr/bin:/bin:/usr/sbin:/sbin:/home/vagrant/bin
# Only required if building the C++ code directly using make - adjust depending on the location of your Git clone
export CPP_SRC_HOME=$HOME/ml-cpp
```

### OS Packages

You need the C++ compiler and the headers for the `zlib` library that comes with the OS.  You also need the archive utilities `unzip` and `bzip2`.  On RHEL/CentOS these can be installed using:

```
sudo yum install bzip2
sudo yum install gcc-c++
sudo yum install unzip
sudo yum install zlib-devel
```

On other Linux distributions the package names are generally the same and you just need to use the correct package manager to install these packages.

### General settings for building the tools

Most of the tools are built via a GNU "configure" script. There are some environment variables that affect the behaviour of this. Therefore, when building ANY tool on Linux, set the following environment variable:

```
export CXX='g++ -std=gnu++0x'
unset LIBRARY_PATH
```

The `CXX` environment variable only needs to be set when building tools on Linux. It should NOT be set when compiling the Machine Learning source code (as this should pick up all settings from our Makefiles).

### gcc

We have to build on old Linux versions to enable our software to run on the older versions of Linux that users have.  However, this means the default compiler on our Linux build servers is also very old.  To enable use of more modern C++ features, we use the default compiler to build a newer version of gcc and then use that to build all our other dependencies.

Download `gcc-6.2.0.tar.bz2` from <http://ftpmirror.gnu.org/gcc/gcc-6.2.0/gcc-6.2.0.tar.bz2>.

Unlike most automake-based tools, gcc must be built in a directory adjacent to the directory containing its source code, so build and install it like this:

```
tar jxvf gcc-6.2.0.tar.bz2
cd gcc-6.2.0
contrib/download_prerequisites
cd ..
mkdir gcc-6.2.0-build
cd gcc-6.2.0-build
../gcc-6.2.0/configure --prefix=/usr/local/gcc62 --enable-languages=c,c++ --with-system-zlib --disable-multilib
make -j 6
sudo make install
```

### Git

Modern versions of Linux will come with Git in their package repositories, and (since we're not redistributing it so don't really care about the exact version used) this is the easiest way to install it. The command will be:

```
sudo yum install git
```

on RHEL clones. **However**, shallow clones do not work correctly before version 1.8.3 of Git, so if the version that yum installs is older you'll still have to build it from scratch. In this case, you may need to uninstall the version that yum installed:

```
git --version
sudo yum remove git
```

If you have to build Git from source in order to get version 1.8.3 or above, this is what to do:

Make sure you install the packages `python-devel`, `curl-devel` and `openssl-devel` using yum or similar before you start this.

Start by running:

```
./configure
```

as usual.

Then run:

```
make prefix=/usr all
sudo make prefix=/usr install
```

Without the `prefix=/usr` bit, you'll end up with a personal Git build in `~/bin` instead of one everyone on the machine can use.

### libxml2

Anonymous FTP to ftp.xmlsoft.org, change directory to libxml2, switch to binary mode, and get `libxml2-2.9.4.tar.gz`.

Uncompress and untar the resulting file. Then run:

```
./configure --prefix=/usr/local/gcc62 --without-python --without-readline
```

This should build an appropriate Makefile. Assuming it does, type:

```
make
sudo make install
```

to install.

### APR

For Linux, before building log4cxx you must download the Apache Portable Runtime (APR), and its utilities from <http://mirrors.sonic.net/apache/apr/apr-1.5.2.tar.bz2>.

Extract the tarball to a temporary directory:

```
tar xvf apr-1.5.2.tar
```

Build using:

```
./configure --prefix=/usr/local/gcc62
make
sudo make install
```

### APR utilities

For Linux, before building log4cxx you must download the Apache Portable Runtime (APR) utilities from <http://mirrors.sonic.net/apache/apr/apr-util-1.5.4.tar.bz2>.

Extract the tarball to a temporary directory:

```
tar xvf apr-util-1.5.4.tar
```

Build using:

```
./configure --prefix=/usr/local/gcc62 --with-apr=/usr/local/gcc62/bin/apr-1-config --with-expat=builtin
make
sudo make install
```

### log4cxx

Download from one of the mirrors listed at <http://www.apache.org/dyn/closer.cgi/logging/log4cxx/0.10.0/apache-log4cxx-0.10.0.tar.gz>.

Unzip using:

```
tar zxvf apache-log4cxx-0.10.0.tar.gz
```

Unfortunately one of the log4cxx headers triggers an annoying (but harmless) g++ warning message. This is due to a copy constructor failing to explicitly call a base class constructor - it doesn't matter as the base class has no member variables, but g++ still complains. You can prevent the header causing a warning (without changing its meaning in any way) by making the changes detailed at <http://issues.apache.org/jira/browse/LOGCXX-314>, i.e. change:

```
#if LOG4CXX_HELGRIND
#define _LOG4CXX_OBJECTPTR_INIT(x) { exchange(x);
#else
#define _LOG4CXX_OBJECTPTR_INIT(x) : p(x) {
#endif
```

to:

```
#if LOG4CXX_HELGRIND
#define _LOG4CXX_OBJECTPTR_INIT(x) : ObjectPtrBase() { exchange(x); 
#else
#define _LOG4CXX_OBJECTPTR_INIT(x) : ObjectPtrBase(), p(x) {
#endif
```

in `src/main/include/log4cxx/helpers/objectptr.h`.

Also, in `src/main/cpp/inputstreamreader.cpp` and `src/main/cpp/socketoutputstream.cpp`, after the last existing `#include` add:

```
#include <string.h>
```

and in `src/examples/cpp/console.cpp`, after the last existing `#include` add:

```
#include <string.h>
#include <stdio.h>
#include <wchar.h>
```

Note that the following 5 edits can be accomplished using these `sed` commands:

```
sed -i -e '152,163s/0x/(char)0x/g' src/main/cpp/locationinfo.cpp
sed -i -e '239,292s/0x/(char)0x/g' src/main/cpp/loggingevent.cpp
sed -i -e '39s/0x/(char)0x/g' src/main/cpp/objectoutputstream.cpp
sed -i -e '84,92s/0x/(char)0x/g' src/main/cpp/objectoutputstream.cpp
sed -i -e '193,214s/0x/(char)0x/g' src/test/cpp/xml/domtestcase.cpp
```

In `src/main/cpp/locationinfo.cpp` replace `0x` with `(char)0x` on lines 152 to 163 - this can be done using the vim command:

```
:152,163s/0x/(char)0x/g
```

In `src/main/cpp/loggingevent.cpp` replace `0x` with `(char)0x` on lines 239 to 292 - this can be done using the vim command:

```
:239,292s/0x/(char)0x/g
```

In `src/main/cpp/objectoutputstream.cpp` replace `0x` with `(char)0x` on lines 39 and 84 to 92 - this can be done using the vim commands:

```
:39s/0x/(char)0x/g
:84,92s/0x/(char)0x/g
```

In `src/test/cpp/xml/domtestcase.cpp` replace `0x` with `(char)0x` on lines 193 to 214 - this can be done using the vim command:

```
:193,214s/0x/(char)0x/g
```

Once all the changes are made, configure using:

```
./configure --prefix=/usr/local/gcc62 --with-charset=utf-8 --with-logchar=utf-8 --with-apr=/usr/local/gcc62 --with-apr-util=/usr/local/gcc62
```

This should build an appropriate Makefile. Assuming it does, type:

```
make
sudo make install
```

to install the necessary headers and libraries.

### Boost 1.65.1

Download version 1.65.1 of Boost from <http://sourceforge.net/projects/boost/files/boost/1.65.1/>. You must get this exact version, as the Machine Learning Makefiles expect it.

Assuming you chose the `.bz2` version, extract it to a temporary directory:

```
bzip2 -cd boost_1_65_1.tar.bz2 | tar xvf -
```

In the resulting `boost_1_65_1` directory, run:

```
./bootstrap.sh cxxflags=-std=gnu++0x --without-libraries=context --without-libraries=coroutine --without-libraries=graph_parallel --without-libraries=log --without-libraries=mpi --without-libraries=python --without-icu
```

This should build the `b2` program, which in turn is used to build Boost.

Edit boost/unordered/detail/implementation.hpp and change line 270 from:

```
    (17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \
```

to:

```
    (3ul)(17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \
```

Then edit `boost/math/tools/config.hpp` and change line 380 from:

```
#if ((defined(__linux__) && !defined(__UCLIBC__) && !defined(BOOST_MATH_HAVE_FIXED_GLIBC)) || defined(__QNX__) || defined(__IBMCPP__)) && !defined(BOOST_NO_FENV_H)
```

to:

```
#if ((!defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS) && defined(__linux__) && !defined(__UCLIBC__) && !defined(BOOST_MATH_HAVE_FIXED_GLIBC)) || defined(__QNX__) || defined(__IBMCPP__)) && !defined(BOOST_NO_FENV_H)
```

Finally, run:

```
./b2 -j6 --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
sudo env PATH="$PATH" ./b2 install --prefix=/usr/local/gcc62 --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
```

to install the Boost headers and libraries.  (Note the `env PATH="$PATH"` bit in the install command - this is because `sudo` usually resets `PATH` and that will cause Boost to rebuild everything again with the default compiler as part of the install!)

### cppunit

Download the latest version of cppunit from <http://dev-www.libreoffice.org/src/cppunit-1.13.2.tar.gz> (or if that no longer exists by the time you read this, find the relevant link on <http://dev-www.libreoffice.org/src>).

Untar it to a temporary directory and run:

```
./configure --prefix=/usr/local/gcc62
```

This should build an appropriate Makefile. Assuming it does, type:

```
make
sudo make install
```

to install the cppunit headers, libraries, binaries and documentation.

### patchelf

Obtain patchelf from <http://nixos.org/releases/patchelf/patchelf-0.9/> - the download file will be `patchelf-0.9.tar.bz2`.

Extract it to a temporary directory using:

```
bzip2 -cd patchelf-0.9.tar.bz2 | tar xvf -
```

In the resulting `patchelf-0.9` directory, run the:

```
./configure --prefix=/usr/local/gcc62
```

script. This should build an appropriate Makefile. Assuming it does, run:

```
make
sudo make install
```

to complete the build.

