# Machine Learning Build Machine Setup for macOS

To ensure everything is consistent for redistributable builds we build all redistributable components from source.

NOTE: if you upgrade macOS then Xcode may need to be re-installed. Please ensure your Xcode is still valid after upgrades.

You will need the following environment variables to be defined:

- `JAVA_HOME` - Should point to the JDK you want to use to run Gradle.
- `CPP_SRC_HOME` - Only required if building the C++ code directly using `make`, as Gradle sets it automatically.
- `PATH` - Must have `/usr/local/bin` before `/usr/bin` and `/bin`.

For example, you might create a `.bashrc` file in your home directory containing something like this:

```
umask 0002
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_121.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
# Only required if building the C++ code directly using make - adjust depending on the location of your Git clone
export CPP_SRC_HOME=$HOME/ml-cpp
```

Note, that bash doesn't read `~/.bashrc` for login shells (which is what you get when you open a new Terminal window) - it reads `~/.bash_profile` or `~/.profile` so you should probably also create a `.bash_profile` file in your home directory which sources `.bashrc`.

### General settings for building the tools

Most of the tools are built via a GNU "configure" script. There are some environment variables that affect the behaviour of this. Therefore, when building ANY tool on macOS, set the following environment variables:

```
export CPP='clang -E'
export CC=clang
export CFLAGS='-O3 -msse4.2'
export CXX='clang++ -std=c++14 -stdlib=libc++'
export CXXFLAGS='-O3 -msse4.2'
export CXXCPP='clang++ -std=c++14 -E'
export LDFLAGS=-Wl,-headerpad_max_install_names
unset CPATH
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH
unset LIBRARY_PATH
```

The above environment variables only need to be set when building tools on macOS. They should NOT be set when compiling the Machine Learning source code (as this should pick up all settings from our Makefiles).

### Xcode

The first major piece of development software to install is Apple's development environment, Xcode, which can be downloaded from <http://developer.apple.com/technologies/xcode.html> . You will need to register as a developer with Apple. Alternatively, you can get the latest version of Xcode from the App Store.

- If you are using Yosemite, you must install Xcode 7.2.x
- If you are using El Capitan, you must install Xcode 8.2.x
- If you are using Sierra, you must install Xcode 9.2.x
- If you are using High Sierra, you must install Xcode 10.1.x
- If you are using Mojave, you must install Xcode 10.3.x

Older versions of Xcode are installed by dragging the app from the `.dmg` file to the `/Applications` directory on your Mac (or if you got it from the App Store it will already be in the `/Applications` directory). More modern versions of Xcode are distributed as a `.xip` file; simply double click the `.xip` file to expand it, then drag `Xcode.app` to your `/Applications` directory.

There are no command line tools out-of-the-box, so you'll need to install them following installation of Xcode. You can do this by running:

```
xcode-select --install
```

at the command prompt.

### Boost 1.71.0

Download version 1.71.0 of Boost from <https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.bz2>. You must get this exact version, as the Machine Learning Makefiles expect it.

Assuming you chose the `.bz2` version, extract it to a temporary directory:

```
bzip2 -cd boost_1_71_0.tar.bz2 | tar xvf -
```

In the resulting `boost_1_71_0` directory, run:

```
./bootstrap.sh --with-toolset=clang --without-libraries=context --without-libraries=coroutine --without-libraries=graph_parallel --without-libraries=mpi --without-libraries=python --without-icu
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

To complete the build, type:

```
./b2 -j8 --layout=versioned --disable-icu cxxflags="-std=c++14 -stdlib=libc++" linkflags="-std=c++14 -stdlib=libc++ -Wl,-headerpad_max_install_names" optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC
sudo ./b2 install --layout=versioned --disable-icu cxxflags="-std=c++14 -stdlib=libc++" linkflags="-std=c++14 -stdlib=libc++ -Wl,-headerpad_max_install_names" optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC
```

to install the Boost headers and libraries.

### cppunit

Download the latest version of cppunit from <http://dev-www.libreoffice.org/src/cppunit-1.13.2.tar.gz> (or if that no longer exists by the time you read this, find the relevant link on <http://dev-www.libreoffice.org/src>).

Assuming you get `cppunit-1.13.2.tar.bz2`, extract it using the commands:

```
bzip2 -cd cppunit-1.13.2.tar.bz2 | tar xvf -
```

In the resulting `cppunit-1.13.2` directory, run the:

```
./configure
```

script. This should build an appropriate Makefile. Assuming it does, type:

```
make
sudo make install
```

to install the cppunit headers, libraries, binaries and documentation.

