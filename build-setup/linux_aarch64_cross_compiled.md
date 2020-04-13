# Machine Learning Build Machine Setup for Linux aarch64 cross compiled

You will need the following environment variables to be defined:

- `JAVA_HOME` - Should point to the JDK you want to use to run Gradle.
- `CPP_CROSS_COMPILE` - Should be set to "aarch64".
- `CPP_SRC_HOME` - Only required if building the C++ code directly using `make`, as Gradle sets it automatically.

For example, you might create a .bashrc file in your home directory containing this:

```
umask 0002
export JAVA_HOME=/usr/local/jdk1.8.0_121
export PATH=$JAVA_HOME/bin:/usr/bin:/bin:/usr/sbin:/sbin
# Only required if building the C++ code directly using make - adjust depending on the location of your Git clone
export CPP_SRC_HOME=$HOME/ml-cpp
export CPP_CROSS_COMPILE=aarch64
```

### Initial Preparation

Start by configuring a native Linux aarch64 build server as described in [linux.md](linux.md).

The remainder of these instructions assume the native aarch64 build server you have configured is for CentOS 7.  This is what builds for distribution are currently built on.

On the fully configured native aarch64 build server, run the following commands:

```
cd /usr
tar jcvf ~/usr-aarch64-linux-gnu.tar.bz2 include lib lib64 local
```

These instructions also assume the host platform is also CentOS 7, but x86_64 instead of aarch64.  It makes life much easier if the host distribution is the same as the target distribution.

Transfer the archive created in your home directory on the native aarch64 build server, `usr-aarch64-linux-gnu.tar.bz2`, to your home directory on the x86_64 host build server.

### OS Packages

You need the C++ compiler and the headers for the `zlib` library that comes with the OS.  You also need the archive utilities `unzip` and `bzip2`.  On RHEL/CentOS these can be installed using:

```
sudo yum install bzip2 gcc-c++ texinfo unzip zlib-devel
```

### Transferred Build Dependencies

Add the dependencies that you copied from the fully configured native aarch64 build server in the "Initial Preparation" step.

```
sudo mkdir -p /usr/local/sysroot-aarch64-linux-gnu/usr
cd /usr/local/sysroot-aarch64-linux-gnu/usr
sudo tar jxvf ~/usr-aarch64-linux-gnu.tar.bz2
cd ..
sudo ln -s usr/lib lib
sudo ln -s usr/lib64 lib64
```

### General settings for building the tools

Most of the tools are built via a GNU "configure" script. There are some environment variables that affect the behaviour of this. Therefore, when building ANY tool on Linux, set the following environment variables:

```
export CFLAGS='-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2'
export CXXFLAGS='-g -O3 -fstack-protector -D_FORTIFY_SOURCE=2'
export LDFLAGS='-Wl,-z,relro -Wl,-z,now'
export LDFLAGS_FOR_TARGET='-Wl,-z,relro -Wl,-z,now'
unset LIBRARY_PATH
```

These environment variables only need to be set when building tools on Linux. They should NOT be set when compiling the Machine Learning source code (as this should pick up all settings from our Makefiles).

### binutils (bootstrap version)

Since we build with a more recent gcc than comes with the host system, we must build it from source.  To build a cross compiler we need cross build tools, so we need to build versions that are compatible with the system compiler that we'll use to build the more recent gcc.

Download `binutils-2.25.tar.bz2` from <http://ftpmirror.gnu.org/binutils/binutils-2.25.tar.bz2>.

Uncompress and untar the resulting file. Then run:

```
unset LD_LIBRARY_PATH
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
./configure --with-sysroot=/usr/local/sysroot-aarch64-linux-gnu --target=aarch64-linux-gnu --with-system-zlib --disable-multilib --disable-libstdcxx
```

This should build an appropriate Makefile. Assuming it does, type:

```
make
sudo make install
```

to install.

### gcc

We have to build on old Linux versions to enable our software to run on the older versions of Linux that users have.  However, this means the default compiler on our Linux build servers is also very old.  To enable use of more modern C++ features, we use the default compiler to build a newer version of gcc and then use that to build all our other dependencies.

Download `gcc-7.5.0.tar.gz` from <http://ftpmirror.gnu.org/gcc/gcc-7.5.0/gcc-7.5.0.tar.gz>.

Unlike most automake-based tools, gcc must be built in a directory adjacent to the directory containing its source code, so build and install it like this:

```
tar zxvf gcc-7.5.0.tar.gz
cd gcc-7.5.0
contrib/download_prerequisites
sed -i -e 's/$(SHLIB_LDFLAGS)/-Wl,-z,relro -Wl,-z,now $(SHLIB_LDFLAGS)/' libgcc/config/t-slibgcc
cd ..
mkdir gcc-7.5.0-build
cd gcc-7.5.0-build
unset LD_LIBRARY_PATH
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin
../gcc-7.5.0/configure --prefix=/usr/local/gcc75 --with-sysroot=/usr/local/sysroot-aarch64-linux-gnu --target=aarch64-linux-gnu --enable-languages=c,c++ --enable-vtable-verify --with-system-zlib --disable-multilib
make -j 6
sudo env PATH="$PATH" make install
```

(Note the `env PATH="$PATH"` bit in the install command - this is because the cross tools we put in `/usr/local/bin` are needed during the install.)

To confirm that everything works correctly run:

```
aarch64-linux-gnu-g++ --version
```

It should print:

```
aarch64-linux-gnu-g++ (GCC) 7.5.0
```

in the first line of the output. If it doesn't then double check that `/usr/local/gcc75/bin` is near the beginning of your `PATH`.

### binutils (final version)

Also due to building on old Linux versions yet wanting to use modern libraries we have to install an up-to-date version of binutils.  This will be used in preference to the bootstrap version by ensuring that `/usr/local/gcc75/bin` is at the beginning of `PATH`.

Download `binutils-2.34.tar.bz2` from <http://ftpmirror.gnu.org/binutils/binutils-2.34.tar.bz2>.

Uncompress and untar the resulting file. Then run:

```
./configure --prefix=/usr/local/gcc75 --with-sysroot=/usr/local/sysroot-aarch64-linux-gnu --target=aarch64-linux-gnu --enable-vtable-verify --with-system-zlib --disable-multilib --disable-libstdcxx --with-gcc-major-version-only
```

This should build an appropriate Makefile. Assuming it does, type:

```
make
sudo make install
```

to install.

### patchelf

Obtain patchelf from <http://nixos.org/releases/patchelf/patchelf-0.9/> - the download file will be `patchelf-0.9.tar.bz2`.

Extract it to a temporary directory using:

```
bzip2 -cd patchelf-0.9.tar.bz2 | tar xvf -
```

In the resulting `patchelf-0.9` directory, run the:

```
./configure --prefix=/usr/local/gcc75
```

script. This should build an appropriate Makefile. Assuming it does, run:

```
make
sudo make install
```

to complete the build.

