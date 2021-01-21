# Machine Learning Build Machine Setup for Linux

These same instructions should work for native compilation on both x86_64 and aarch64 architectures.

To ensure everything is consistent for redistributable builds we build all redistributable components from source with a specific version of gcc.

You will need the following environment variables to be defined:

- `JAVA_HOME` - Should point to the JDK you want to use to run Gradle.
- `CPP_SRC_HOME` - Only required if building the C++ code directly using `make`, as Gradle sets it automatically.
- `PATH` - Must have `/usr/local/gcc93/bin` before `/usr/bin` and `/bin`.
- `LD_LIBRARY_PATH` - Must have `/usr/local/gcc93/lib64` and `/usr/local/gcc93/lib` before `/usr/lib` and `/lib`.

For example, you might create a `.bashrc` file in your home directory containing something like this:

```
umask 0002
export JAVA_HOME=/usr/local/jdk1.8.0_121
export LD_LIBRARY_PATH=/usr/local/gcc93/lib64:/usr/local/gcc93/lib:/usr/lib:/lib
export PATH=$JAVA_HOME/bin:/usr/local/gcc93/bin:/usr/local/cmake/bin:/usr/bin:/bin:/usr/sbin:/sbin:/home/vagrant/bin
# Only required if building the C++ code directly using make - adjust depending on the location of your Git clone
export CPP_SRC_HOME=$HOME/ml-cpp
```

### OS Packages

You need the C++ compiler and the headers for the `zlib` library that comes with the OS.  You also need the archive utilities `unzip` and `bzip2`. `libffi-devel` and `openssl-devel` are dependencies for building PyTorch. Finally, the unit tests for date/time parsing require the `tzdata` package that contains the Linux timezone database.  On RHEL/CentOS these can be installed using:

```
sudo yum install bzip2 gcc-c++ libffi-devel openssl-devel texinfo tzdata unzip zlib-devel
```

On other Linux distributions the package names are generally the same and you just need to use the correct package manager to install these packages.

### General settings for building the tools

Most of the tools are built via a GNU "configure" script. There are some environment variables that affect the behaviour of this. Therefore, when building ANY tool on Linux, set the following environment variables:

```
export CFLAGS='-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse'
export CXX='g++ -std=gnu++17'
export CXXFLAGS='-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2 -msse4.2 -mfpmath=sse'
export LDFLAGS='-Wl,-z,relro -Wl,-z,now'
export LDFLAGS_FOR_TARGET='-Wl,-z,relro -Wl,-z,now'
unset LIBRARY_PATH
```

For aarch64 replace `-msse4.2 -mfpmath=sse` with `-march=armv8-a+crc+crypto`.

These environment variables only need to be set when building tools on Linux. They should NOT be set when compiling the Machine Learning source code (as this should pick up all settings from our Makefiles).

### gcc

We have to build on old Linux versions to enable our software to run on the older versions of Linux that users have.  However, this means the default compiler on our Linux build servers is also very old.  To enable use of more modern C++ features, we use the default compiler to build a newer version of gcc and then use that to build all our other dependencies.

Download `gcc-9.3.0.tar.gz` from <http://ftpmirror.gnu.org/gcc/gcc-9.3.0/gcc-9.3.0.tar.gz>.

Unlike most automake-based tools, gcc must be built in a directory adjacent to the directory containing its source code, so build and install it like this:

```
tar zxvf gcc-9.3.0.tar.gz
cd gcc-9.3.0
contrib/download_prerequisites
sed -i -e 's/$(SHLIB_LDFLAGS)/-Wl,-z,relro -Wl,-z,now $(SHLIB_LDFLAGS)/' libgcc/config/t-slibgcc
cd ..
mkdir gcc-9.3.0-build
cd gcc-9.3.0-build
unset CXX
unset LD_LIBRARY_PATH
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
../gcc-9.3.0/configure --prefix=/usr/local/gcc93 --enable-languages=c,c++ --enable-vtable-verify --with-system-zlib --disable-multilib
make -j 6
sudo make install
```

It's important that gcc itself is built using the system compiler in C++98 mode, hence the adjustment to `PATH` and unsetting of `CXX` and `LD_LIBRARY_PATH`.

After the gcc build is complete, if you are going to work through the rest of these instructions in the same shell remember to reset the `CXX` environment variable so that the remaining C++ components get built with C++17:

```
export CXX='g++ -std=gnu++17'
```

To confirm that everything works correctly run:

```
g++ --version
```

It should print:

```
g++ (GCC) 9.3.0
```

in the first line of the output. If it doesn't then double check that `/usr/local/gcc93/bin` is near the beginning of your `PATH`.

### binutils

Also due to building on old Linux versions yet wanting to use modern libraries we have to install an up-to-date version of binutils.

Download `binutils-2.34.tar.bz2` from <http://ftpmirror.gnu.org/binutils/binutils-2.34.tar.bz2>.

Uncompress and untar the resulting file. Then run:

```
./configure --prefix=/usr/local/gcc93 --enable-vtable-verify --with-system-zlib --disable-libstdcxx --with-gcc-major-version-only
```

This should build an appropriate Makefile. Assuming it does, type:

```
make
sudo make install
```

to install.

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

Anonymous FTP to ftp.xmlsoft.org, change directory to libxml2, switch to binary mode, and get `libxml2-2.9.7.tar.gz`.

Uncompress and untar the resulting file. Then run:

```
./configure --prefix=/usr/local/gcc93 --without-python --without-readline
```

This should build an appropriate Makefile. Assuming it does, type:

```
make
sudo make install
```

to install.

### Boost 1.71.0

Download version 1.71.0 of Boost from <https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.bz2>. You must get this exact version, as the Machine Learning Makefiles expect it.

Assuming you chose the `.bz2` version, extract it to a temporary directory:

```
bzip2 -cd boost_1_71_0.tar.bz2 | tar xvf -
```

In the resulting `boost_1_71_0` directory, run:

```
./bootstrap.sh --without-libraries=context --without-libraries=coroutine --without-libraries=graph_parallel --without-libraries=mpi --without-libraries=python --without-icu
```

This should build the `b2` program, which in turn is used to build Boost.

Edit `boost/unordered/detail/implementation.hpp` and change line 287 from:

```
    (17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \
```

to:

```
    (3ul)(17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \
```

Finally, run:

```
./b2 -j6 --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC define=_FORTIFY_SOURCE=2 cxxflags='-std=gnu++17 -fstack-protector -msse4.2 -mfpmath=sse' linkflags='-std=gnu++17 -Wl,-z,relro -Wl,-z,now'
sudo env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" ./b2 install --prefix=/usr/local/gcc93 --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC define=_FORTIFY_SOURCE=2 cxxflags='-std=gnu++17 -fstack-protector -msse4.2 -mfpmath=sse' linkflags='-std=gnu++17 -Wl,-z,relro -Wl,-z,now'
```

to install the Boost headers and libraries.  (Note the `env PATH="$PATH"` bit in the install command - this is because `sudo` usually resets `PATH` and that will cause Boost to rebuild everything again with the default compiler as part of the install!)

For aarch64 replace `-msse4.2 -mfpmath=sse` with `-march=armv8-a+crc+crypto`.

### patchelf

Obtain patchelf from <http://nixos.org/releases/patchelf/patchelf-0.10/> - the download file will be `patchelf-0.10.tar.bz2`.

Extract it to a temporary directory using:

```
bzip2 -cd patchelf-0.10.tar.bz2 | tar xvf -
```

In the resulting `patchelf-0.10` directory, run the:

```
./configure --prefix=/usr/local/gcc93
```

script. This should build an appropriate Makefile. Assuming it does, run:

```
make
sudo make install
```

to complete the build.

### CMake

CMake version 3.5 is the minimum required to build PyTorch. Download version 3.19.3 from <https://github.com/Kitware/CMake/releases/download/v3.19.3/cmake-3.19.3-Linux-x86_64.sh> and install:

```
chmod +x cmake-3.19.3-Linux-x86_64.sh
sudo mkdir /usr/local/cmake
sudo ./cmake-3.19.3-Linux-x86_64.sh --skip-license --prefix=/usr/local/cmake
```  

Please ensure `/usr/local/cmake/bin` is in your `PATH` environment variable. 

### Python 

PyTorch currently requires Python 3.6, 3.7 or 3.8, and version 3.7 appears to cause fewest problems in their test status matrix, so we use that. If your system does not have a requisite version of Python install it with a package manager or build the last 3.7 release from source by downloading `Python-3.7.9.tgz` from <https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz> then extract and build:

```
tar -xzf Python-3.7.9.tgz
cd Python-3.7.9/
./configure --enable-optimizations
make
sudo make altinstall
```

### PyTorch 1.7.1

PyTorch requires that certain Python modules are installed. Install these modules with `pip` using the same Python version you will build PyTorch with. If you followed the instructions above and built Python from source use `python3.7`:

```
sudo /usr/local/bin/python3.7 -m pip install install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
```

Then obtain the PyTorch code:
```
git clone --depth=1 --branch=v1.7.1 git@github.com:pytorch/pytorch.git
cd pytorch
git submodule sync
git submodule update --init --recursive
```

Build as follows:
```
export BUILD_TEST=OFF
export BUILD_CAFFE2=OFF
export USE_NUMPY=OFF
export USE_DISTRIBUTED=OFF
/usr/local/bin/python3.7 setup.py install
```

Once built copy headers and libraries to system directories:

```
sudo mkdir /usr/local/gcc93/include/pytorch
sudo cp -r torch/include/* /usr/local/gcc93/include/pytorch/ 
sudo cp torch/lib/libtorch_cpu.so /usr/local/gcc93/lib 
sudo cp torch/lib/libc10.so /usr/local/gcc93/lib 
```

### valgrind

`valgrind` is not required to build the code.  However, since we build `gcc` ourselves, if you want
to debug with `valgrind` then you'll get better results if you build a version that's compatible
with our `gcc` instead of using the version you can get via your package manager.

If you find yourself needing to do this, download `valgrind` from <http://valgrind.org/downloads/> - the
download file will be `valgrind-3.15.0.tar.bz2`.

Extract it to a temporary directory using:

```
tar jxvf valgrind-3.15.0.tar.bz2
```

In the resulting `valgrind-3.15.0` directory, run:

```
unset CFLAGS
unset CXXFLAGS
./configure --prefix=/usr/local/gcc93 --disable-dependency-tracking --enable-only64bit
```

The reason for unsetting the compiler flags is that `valgrind` does not build correctly
with the fortified options we have to use for libraries we ship.

This should build an appropriate Makefile. Assuming it does, run:

```
make
sudo make install
```

to complete the build.

