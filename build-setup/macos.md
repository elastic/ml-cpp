# Machine Learning Build Machine Setup for macOS

These same instructions should work for native compilation on both x86_64 and aarch64 architectures.

To ensure everything is consistent for redistributable builds we build all redistributable components from source.

NOTE: if you upgrade macOS then Xcode may need to be re-installed. Please ensure your Xcode is still valid after upgrades.

You will need the following environment variables to be defined:

- `JAVA_HOME` - Should point to the JDK you want to use to run Gradle.
- `CPP_SRC_HOME` - Only required if building the C++ code directly using `cmake`, as Gradle sets it automatically.
- `PATH` - Must have `/usr/local/bin` before `/usr/bin` and `/bin`.

For example, you might create a `.bashrc` file in your home directory containing something like this:

```
umask 0002
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_121.jdk/Contents/Home
export PYTHONHOME=/Library/Frameworks/Python.framework/Versions/3.10
export PATH=$JAVA_HOME/bin:$PYTHONHOME/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
# Only required if building the C++ code directly using cmake - adjust depending on the location of your Git clone
export CPP_SRC_HOME=$HOME/ml-cpp
```

Note, that bash doesn't read `~/.bashrc` for login shells (which is what you get when you open a new Terminal window) - it reads `~/.bash_profile` or `~/.profile` so you should probably also create a `.bash_profile` file in your home directory which sources `.bashrc`.

### General settings for building the tools

Some tools may be built via a GNU "configure" script. There are some environment variables that affect the behaviour of this. Therefore, when building ANY tool on macOS, set the following environment variables:

```
export SSEFLAGS=`[ $(uname -m) = x86_64 ] && echo -msse4.2`
export CPP='clang -E'
export CC=clang
export CFLAGS="-O3 $SSEFLAGS"
export CXX='clang++ -std=c++17 -stdlib=libc++'
export CXXFLAGS="-O3 $SSEFLAGS"
export CXXCPP='clang++ -std=c++17 -E'
export LDFLAGS=-Wl,-headerpad_max_install_names
unset CPATH
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH
unset LIBRARY_PATH
```

The above environment variables only need to be set when building tools on macOS. They should NOT be set when compiling the Machine Learning source code (as this should pick up all settings from our build system).

### Xcode

The first major piece of development software to install is Apple's development environment, Xcode, which can be downloaded from <https://developer.apple.com/download/> . You will need to register as a developer with Apple. Alternatively, you can get the latest version of Xcode from the App Store.

For C++17 Xcode 10 is required, and this requires macOS High Sierra or above. Therefore you must be running macOS High Sierra (10.13) or above to build the Machine Learning C++ code. The binary download of Python 3.10 requires macOS Big Sur or above, so to follow the instructions in this page to the letter you'll need macOS Big Sur (11) or above.

- If you are using High Sierra, you must install Xcode 10.1.x
- If you are using Mojave, you must install Xcode 11.3.x
- If you are using Catalina, you must install Xcode 12.4.x
- If you are using Big Sur, you must install Xcode 13.2.x
- If you are using Monterey or Ventura, you must install Xcode 14.2.x or above

Xcode is distributed as a `.xip` file; simply double click the `.xip` file to expand it, then drag `Xcode.app` to your `/Applications` directory.
(Older versions of Xcode can be downloaded from [here](https://developer.apple.com/download/more/), provided you are signed in with your Apple ID.)

There are no command line tools out-of-the-box, so you'll need to install them following installation of Xcode. You can do this by running:

```
xcode-select --install
```

at the command prompt.

### Boost 1.83.0

Download version 1.83.0 of Boost from <https://boostorg.jfrog.io/artifactory/main/release/1.83.0/source/boost_1_83_0.tar.bz2>. You must get this exact version, as the Machine Learning build system requires it.

Assuming you chose the `.bz2` version, extract it to a temporary directory:

```
bzip2 -cd boost_1_83_0.tar.bz2 | tar xvf -
```

In the resulting `boost_1_83_0` directory, run:

```
./bootstrap.sh --with-toolset=clang --without-libraries=context --without-libraries=coroutine --without-libraries=graph_parallel --without-libraries=mpi --without-libraries=python --without-icu
```

This should build the `b2` program, which in turn is used to build Boost.

Edit `boost/unordered/detail/prime_fmod.hpp` and change line 134 from

```
    (13ul)(29ul)(53ul)(97ul)(193ul)(389ul)(769ul)(1543ul)(3079ul)(6151ul)(       \
```

to:

```
    (3ul)(13ul)(29ul)(53ul)(97ul)(193ul)(389ul)(769ul)(1543ul)(3079ul)(6151ul)(       \
```


To complete the build, type:

```
./b2 -j8 --layout=versioned --disable-icu cxxflags="-std=c++17 -stdlib=libc++ $SSEFLAGS" linkflags="-std=c++17 -stdlib=libc++ -Wl,-headerpad_max_install_names" optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC
sudo ./b2 install --layout=versioned --disable-icu cxxflags="-std=c++17 -stdlib=libc++ $SSEFLAGS" linkflags="-std=c++17 -stdlib=libc++ -Wl,-headerpad_max_install_names" optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC
```

to install the Boost headers and libraries.

### CMake

CMake version 3.19.2 is the minimum required to build ml-cpp.  Download the graphical installer for version 3.23.2 from <https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2-macos-universal.dmg> (or get a more recent version).

Open the `.dmg` and install the application it by dragging it to the `Applications` folder.

Then make the `cmake` program accessible to programs that look in `/usr/local/bin`:

```
sudo mkdir -p /usr/local/bin
sudo ln -s /Applications/CMake.app/Contents/bin/cmake /usr/local/bin/cmake
```

### Python 3.10

PyTorch currently requires Python 3.7 or higher; we use version 3.10.

Download the graphical installer for Python 3.10.9 from <https://www.python.org/ftp/python/3.10.9/python-3.10.9-macos11.pkg>.

Install using all the default options.  When the installer completes a Finder window pops up.  Double click the `Install Certificates.command` file in this folder to install the SSL certificates Python needs.

### PyTorch 1.13.1

PyTorch requires that certain Python modules are installed.  To install them:

```
sudo /Library/Frameworks/Python.framework/Versions/3.10/bin/pip3.10 install install numpy ninja pyyaml setuptools cffi typing_extensions future six requests dataclasses
```

Then obtain the PyTorch code:

```
git clone --depth=1 --branch=v1.13.1 https://github.com/pytorch/pytorch.git
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

Edit `tools/setup_helpers/cmake.py` and add `"DNNL_TARGET_ARCH"` to the list
of environment variables that get passed through to CMake (around line 215).

Build as follows:

```
export BLAS=vecLib
export BUILD_TEST=OFF
export BUILD_CAFFE2=OFF
export USE_NUMPY=OFF
export USE_DISTRIBUTED=OFF
[ $(uname -m) != x86_64 ] && export DNNL_TARGET_ARCH=AARCH64
export USE_MKLDNN=ON
export USE_QNNPACK=OFF
export USE_PYTORCH_QNNPACK=OFF
[ $(uname -m) = x86_64 ] && export USE_XNNPACK=OFF
export PYTORCH_BUILD_VERSION=1.13.1
export PYTORCH_BUILD_NUMBER=1
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 setup.py install
```

Once built copy headers and libraries to system directories:

```
sudo mkdir -p /usr/local/include/pytorch
sudo cp -r torch/include/* /usr/local/include/pytorch/
sudo cp torch/lib/libtorch_cpu.dylib /usr/local/lib/
sudo cp torch/lib/libc10.dylib /usr/local/lib/
```

