# Machine Learning Build Machine Setup for macOS cross compiled on Linux

You will need the following environment variables to be defined:

- `JAVA_HOME` - Should point to the JDK you want to use to run Gradle.
- `CPP_CROSS_COMPILE` - Should be set to "macosx".
- `CPP_SRC_HOME` - Only required if building the C++ code directly using `make`, as Gradle sets it automatically.

For example, you might create a .bashrc file in your home directory containing this:

```
umask 0002
export JAVA_HOME=/usr/local/jdk1.8.0_121
export PATH=$JAVA_HOME/bin:/usr/bin:/bin:/usr/sbin:/sbin
# Only required if building the C++ code directly using make - adjust depending on the location of your Git clone
export CPP_SRC_HOME=$HOME/ml-cpp
export CPP_CROSS_COMPILE=macosx
```

### Initial Preparation

Start by configuring a native macOS build server as described in [macos.md](macos.md).

The remainder of these instructions assume the macOS build server you have configured is for macOS 10.13 (High Sierra).  This is what builds for distribution are currently built on.

On the fully configured macOS build server, run the following commands:

```
cd /usr
tar jcvf ~/usr-x86_64-apple-macosx10.13.tar.bz2 include lib local
cd /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr
tar jcvf ~/xcode-x86_64-apple-macosx10.13.tar.bz2 include
```

These instructions also assume the host platform is Ubuntu 18.04.  It makes life much easier if the host platform is a version of Ubuntu that's new enough to run the official binary distribution of clang/LLVM (otherwise it would be necessary to build clang/LLVM from source).

Transfer the two archives created in your home directory on the macOS build server, `usr-x86_64-apple-macosx10.13.tar.bz2` and `xcode-x86_64-apple-macosx10.13.tar.bz2`, to your home directory on the cross compilation host build server.

### OS Packages

You need clang 6.0, plus a number of other build tools.  They can be installed on modern Ubuntu as follows:

```
sudo apt-get install automake autogen build-essential bzip2 git gobjc libtool software-properties-common unzip wget

wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt-add-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic main"
sudo apt-get install clang-6.0 clang-6.0-doc libclang1-6.0 libllvm6.0 lldb-6.0 llvm-6.0 llvm-6.0-doc llvm-6.0-runtime
```

(It is strongly recommended NOT to attempt to create a cross compile environment on an old version of Linux, because you will have to build clang from source and you need a modern C++ compiler to build clang.  So you would probably end up first building a modern version of gcc using the system default gcc, then building clang using the modern gcc.)

### Transferred Build Dependencies

Add the dependencies that you copied from the fully configured macOS build server in the "Initial Preparation" step.

```
sudo mkdir -p /usr/local/sysroot-x86_64-apple-macosx10.13/usr
cd /usr/local/sysroot-x86_64-apple-macosx10.13/usr
sudo tar jxvf ~/usr-x86_64-apple-macosx10.13.tar.bz2
sudo tar jxvf ~/xcode-x86_64-apple-macosx10.13.tar.bz2
```

### cctools-port

You need to obtain Linux ports of several Apple development tools.  The easiest way to get them is to use the [cctools-port project on GitHub](https://github.com/tpoechtrager/cctools-port):

```
git clone https://github.com/tpoechtrager/cctools-port.git
cd cctools-port/cctools
git checkout 921-ld64-409.12
sed -i -e 's/autoconf/autoreconf -fi/' autogen.sh
export CC=clang-6.0
export CXX=clang++-6.0
./autogen.sh
./configure --target=x86_64-apple-macosx10.13 --with-llvm-config=/usr/bin/llvm-config-6.0
make
sudo make install
```

The "921-ld64-409.12" branch in the [cctools-port repository](https://github.com/tpoechtrager/cctools-port) corresponds to the tools for macOS 10.13 High Sierra and clang 6.0.  (A different branch would be required for newer versions of the OS/compiler.)

