# Machine Learning Build Machine Setup for Linux

These same instructions should work for native compilation on both x86_64 and aarch64 architectures.

To ensure everything is consistent for redistributable builds we build all redistributable components from source with a specific version of gcc.

You will need the following environment variables to be defined:

- `JAVA_HOME` - Should point to the JDK you want to use to run Gradle.
- `CPP_SRC_HOME` - Only required if building the C++ code directly using `cmake`, as Gradle sets it automatically.
- `PATH` - Must have `/usr/local/gcc103/bin` before `/usr/bin` and `/bin`.
- `LD_LIBRARY_PATH` - Must have `/usr/local/gcc103/lib64` and `/usr/local/gcc103/lib` before `/usr/lib` and `/lib`.

For example, you might create a `.bashrc` file in your home directory containing something like this:

```
umask 0002
export JAVA_HOME=/usr/local/jdk1.8.0_121
export LD_LIBRARY_PATH=/usr/local/gcc103/lib64:/usr/local/gcc103/lib:/usr/lib:/lib
export PATH=$JAVA_HOME/bin:/usr/local/gcc103/bin:/usr/local/cmake/bin:/usr/bin:/bin:/usr/sbin:/sbin:/home/vagrant/bin
# Only required if building the C++ code directly using cmake - adjust depending on the location of your Git clone
export CPP_SRC_HOME=$HOME/ml-cpp
```

### OS Packages

You need the C++ compiler and the headers for the `zlib` library that comes with the OS.  You also need the archive utilities `unzip`, `bzip2` and `xz`. `libffi-devel` and `openssl-devel` are dependencies for building PyTorch. Finally, the unit tests for date/time parsing require the `tzdata` package that contains the Linux timezone database.  On RHEL/CentOS these can be installed using:

```
sudo yum install bzip2 gcc-c++ libffi-devel openssl-devel texinfo tzdata unzip xz zlib-devel
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

These environment variables only need to be set when building tools on Linux. They should NOT be set when compiling the Machine Learning source code (as this should pick up all settings from our build system).

### gcc

We have to build on old Linux versions to enable our software to run on the older versions of Linux that users have.  However, this means the default compiler on our Linux build servers is also very old.  To enable use of more modern C++ features, we use the default compiler to build a newer version of gcc and then use that to build all our other dependencies.

Download `gcc-10.3.0.tar.gz` from <http://ftpmirror.gnu.org/gcc/gcc-10.3.0/gcc-10.3.0.tar.gz>.

Unlike most automake-based tools, gcc must be built in a directory adjacent to the directory containing its source code, so build and install it like this:

```
tar zxvf gcc-10.3.0.tar.gz
cd gcc-10.3.0
contrib/download_prerequisites
sed -i -e 's/$(SHLIB_LDFLAGS)/-Wl,-z,relro -Wl,-z,now $(SHLIB_LDFLAGS)/' libgcc/config/t-slibgcc
cd ..
mkdir gcc-10.3.0-build
cd gcc-10.3.0-build
unset CXX
unset LD_LIBRARY_PATH
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
../gcc-10.3.0/configure --prefix=/usr/local/gcc103 --enable-languages=c,c++ --enable-vtable-verify --with-system-zlib --disable-multilib
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
g++ (GCC) 10.3.0
```

in the first line of the output. If it doesn't then double check that `/usr/local/gcc103/bin` is near the beginning of your `PATH`.

### binutils

Also due to building on old Linux versions yet wanting to use modern libraries we have to install an up-to-date version of binutils.

Download `binutils-2.37.tar.bz2` from <http://ftpmirror.gnu.org/binutils/binutils-2.37.tar.bz2>.

Uncompress and untar the resulting file. Then run:

```
./configure --prefix=/usr/local/gcc103 --enable-vtable-verify --with-system-zlib --disable-libstdcxx --with-gcc-major-version-only
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

Download `libxml2-2.9.14.tar.xz` from <https://download.gnome.org/sources/libxml2/2.9/libxml2-2.9.14.tar.xz>.

Uncompress and untar the resulting file. Then run:

```
./configure --prefix=/usr/local/gcc103 --without-python --without-readline
```

This should build an appropriate Makefile. Assuming it does, type:

```
make
sudo make install
```

to install.

### Boost 1.77.0

Download version 1.77.0 of Boost from <https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.bz2>. You must get this exact version, as the Machine Learning build system requires it.

Assuming you chose the `.bz2` version, extract it to a temporary directory:

```
bzip2 -cd boost_1_77_0.tar.bz2 | tar xvf -
```

In the resulting `boost_1_77_0` directory, run:

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
sudo env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" ./b2 install --prefix=/usr/local/gcc103 --layout=versioned --disable-icu pch=off optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC define=_FORTIFY_SOURCE=2 cxxflags='-std=gnu++17 -fstack-protector -msse4.2 -mfpmath=sse' linkflags='-std=gnu++17 -Wl,-z,relro -Wl,-z,now'
```

to install the Boost headers and libraries.  (Note the `env PATH="$PATH"` bit in the install command - this is because `sudo` usually resets `PATH` and that will cause Boost to rebuild everything again with the default compiler as part of the install!)

For aarch64 replace `-msse4.2 -mfpmath=sse` with `-march=armv8-a+crc+crypto`.

### patchelf

Obtain patchelf from <https://github.com/NixOS/patchelf/releases/download/0.13/patchelf-0.13.tar.bz2>.

Extract it to a temporary directory using:

```
bzip2 -cd patchelf-0.13.tar.bz2 | tar xvf -
```

In the resulting `patchelf-0.13.20210805.a949ff2` directory, run the:

```
./configure --prefix=/usr/local/gcc103
```

script. This should build an appropriate Makefile. Assuming it does, run:

```
make
sudo make install
```

to complete the build.

### CMake

CMake version 3.19.2 is the minimum required to build ml-cpp. Download version 3.23.2 from <https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2-Linux-x86_64.sh> and install:

```
chmod +x cmake-3.23.2-Linux-x86_64.sh
sudo mkdir /usr/local/cmake
sudo ./cmake-3.23.2-Linux-x86_64.sh --skip-license --prefix=/usr/local/cmake
```

Please ensure `/usr/local/cmake/bin` is in your `PATH` environment variable.

### OpenSSL

Python 3.10 requires OpenSSL 1.1.1. No other version is acceptable.

If the `openssl-devel` package for your distribution happens to be version 1.1.1 then you can skip this step. Otherwise, you need to build OpenSSL 1.1.1 from source.

Download `openssl-1.1.1q.tar.gz` from <https://www.openssl.org/source/old/1.1.1/openssl-1.1.1q.tar.gz>, then build as follows:

```
tar zxvf openssl-1.1.1q.tar.gz
cd openssl-1.1.1q
./Configure --prefix=/usr/local/gcc103 shared linux-`uname -m`
make
make install
```

### Python 3.10

PyTorch currently requires Python 3.7 or higher; we use version 3.10. If your system does not have a requisite version of Python install it with a package manager or build the last 3.10 release from source by downloading `Python-3.10.9.tgz` from <https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz> then extract as follows:

```
tar xzf Python-3.10.9.tgz
cd Python-3.10.9
```

If the distribution you are building on uses OpenSSL 1.1.1 as its built in OpenSSL version then configure as follows:

```
./configure --prefix=/usr/local/gcc103 --enable-optimizations
```

If you had to build OpenSSL 1.1.1 yourself then on x86_64 configure like this:

```
sed -i -e 's~ssldir/lib~ssldir/lib64~' configure
./configure --prefix=/usr/local/gcc103 --enable-optimizations --with-openssl=/usr/local/gcc103 --with-openssl-rpath=/usr/local/gcc103/lib64
```

or on aarch64 configure like this:

```
./configure --prefix=/usr/local/gcc103 --enable-optimizations --with-openssl=/usr/local/gcc103 --with-openssl-rpath=/usr/local/gcc103/lib
```

Finally, build as follows:

```
make
sudo make altinstall
```

### Intel MKL

Skip this step if you are building on aarch64.

Intel Maths Kernel Library is an optimized BLAS library which
greatly improves the performance of PyTorch CPU inference 
when PyTorch is built with it. 

```
sudo yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
sudo yum -y install intel-mkl-2020.4-912
```

Then copy the shared libraries to the system directory:

```
sudo cp /opt/intel/mkl/lib/intel64/libmkl*.so /usr/local/gcc103/lib
```

### PyTorch 1.13.1

(This step requires a reasonable amount of memory. It failed on a machine with 8GB of RAM. It succeeded on a 16GB machine. You can specify the number of parallel jobs using environment variable MAX_JOBS. Lower number of jobs will reduce memory usage.)

PyTorch requires that certain Python modules are installed. Install these modules with `pip` using the same Python version you will build PyTorch with. If you followed the instructions above and built Python from source use `python3.10`:

```
sudo /usr/local/gcc103/bin/python3.10 -m pip install install numpy ninja pyyaml setuptools cffi typing_extensions future six requests dataclasses
```

For aarch64 the `ninja` module is not available, so use:

```
sudo /usr/local/gcc103/bin/python3.10 -m pip install install numpy pyyaml setuptools cffi typing_extensions future six requests dataclasses
```

Then obtain the PyTorch code:

```
git clone --depth=1 --branch=v1.13.1 git@github.com:pytorch/pytorch.git
cd pytorch
git submodule sync
git submodule update --init --recursive
```

Edit `torch/csrc/jit/codegen/fuser/cpu/fused_kernel.cpp` and replace all
occurrences of `system(` with `strlen(`. This file is used to compile
fused CPU kernels, which we do not expect to be doing and never want to
do for security reasons. Replacing the calls to `system()` ensures that
a heuristic virus scanner looking for potentially dangerous function
calls in our shipped product will not encounter these functions that run
external processes.

Build as follows:

```
[ $(uname -m) = x86_64 ] && export BLAS=MKL || export BLAS=Eigen
export BUILD_TEST=OFF
[ $(uname -m) = x86_64 ] && export BUILD_CAFFE2=OFF
[ $(uname -m) != x86_64 ] && export USE_FBGEMM=OFF
[ $(uname -m) != x86_64 ] && export USE_KINETO=OFF
[ $(uname -m) = x86_64 ] && export USE_NUMPY=OFF
export USE_DISTRIBUTED=OFF
export USE_MKLDNN=ON
export USE_QNNPACK=OFF
export USE_PYTORCH_QNNPACK=OFF
[ $(uname -m) = x86_64 ] && export USE_XNNPACK=OFF
export PYTORCH_BUILD_VERSION=1.13.1
export PYTORCH_BUILD_NUMBER=1
/usr/local/gcc103/bin/python3.10 setup.py install
```

Once built copy headers and libraries to system directories:

```
sudo mkdir -p /usr/local/gcc103/include/pytorch
sudo cp -r torch/include/* /usr/local/gcc103/include/pytorch/
sudo cp torch/lib/libtorch_cpu.so /usr/local/gcc103/lib
sudo cp torch/lib/libc10.so /usr/local/gcc103/lib
```

### Intel Extension for PyTorch
If you are building on x86_64, you can optionally build the Intel Extension for PyTorch (IPEX). This extension provides additional optimizations for Intel CPUs. It is not available for aarch64. IPEX library is required to run PyTorch models quantized using the IPEX backend.

Begin by cloning the IPEX repository:

```bash
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git fetch --all
git checkout v1.13.100+cpu
git submodule sync
git submodule update --init --recursive
git config --global --add safe.directory `pwd`
```

IPEX expects that PyTorch build directory contains a file `build-version`, which contains the PyTorch version string. This file is not created by the PyTorch build process, so we need to create it manually:
```bash
 echo "1.13.1+cpu\n" > ${PYTORCH_SRC_DIR}/torch/build-version
```

This assumes that you have cloned PyTorch in the directory `${PYTORCH_SRC_DIR}` in the step above. Make sure that this path is correct.

Building IPEX requires a lot of memory. To reduce this requirement, we can patch the IPEX build system to lower the number of parallel processes. We use `sed` to replace the call to `multiprocessing.cpu_count()` with a constant `1` in the file `setup.py`:
```bash
sed -i 's/multiprocessing.cpu_count()/1/g' setup.py
```

Finally, we can build IPEX:
```bash
export CC=/usr/local/gcc103/bin/gcc
export TORCH_VERSION="v1.13.1"
export TORCH_IPEX_VERSION="1.13.100+cpu"
/usr/local/gcc103/bin/python3.10 -m pip install -r requirements.txt
/usr/local/gcc103/bin/python3.10 setup.py clean
/usr/local/gcc103/bin/python3.10 setup.py build_clib ${PYTORCH_SRC_DIR}/torch
cp build/Release/packages/intel_extension_for_pytorch/lib/libintel-ext-pt-cpu.so /usr/local/gcc103/lib
```

### valgrind

`valgrind` is not required to build the code.  However, since we build `gcc` ourselves, if you want
to debug with `valgrind` then you'll get better results if you build a version that's compatible
with our `gcc` instead of using the version you can get via your package manager.

If you find yourself needing to do this, download `valgrind` from <http://valgrind.org/downloads/> - the
download file will be `valgrind-3.17.0.tar.bz2`.

Extract it to a temporary directory using:

```
tar jxvf valgrind-3.17.0.tar.bz2
```

In the resulting `valgrind-3.17.0` directory, run:

```
unset CFLAGS
unset CXXFLAGS
./configure --prefix=/usr/local/gcc103 --disable-dependency-tracking --enable-only64bit
```

The reason for unsetting the compiler flags is that `valgrind` does not build correctly
with the fortified options we have to use for libraries we ship.

This should build an appropriate Makefile. Assuming it does, run:

```
make
sudo make install
```

to complete the build.

