# Machine Learning Build Machine Setup for Windows

These instructions are for setting up a Windows build machine from scratch.

For subsequent machines, the setup can be simplified by copying components from the first machine - see [windows_simple.md](windows_simple.md).

The build system assumes Windows itself is installed on the `C:` drive. If this is not the case on your machine, the Makefiles will not work.

On Windows, the build system runs within Git for Windows. This provides a bash shell that enables our Unix build system to work on Windows without alteration.

If you use Gradle to build Machine Learning you do not need to work in Git Bash, as Gradle tasks will run it when necessary.  In order to develop interactively on Windows, you must work in Git Bash.  You will need the following environment variables to be defined:

- `CPP_SRC_HOME` - This must contain the absolute path of the top level source code directory.

You will also need to add several directories to your `PATH` environment variable. These need to be in the MinGW format, i.e. beginning with /drive letter/ followed by the directory path using forward slashes instead of backslashes, and with no spaces in the directory names. Where a Windows directory name contains a space, you must use the 8.3 version of that directory name (which you can find using `dir /x` at a command prompt). The directories that need to be added are:

- `/c/PROGRA~2/MICROS~2/2019/Professional/VC/Tools/MSVC/14.23.28105/bin/Hostx64/x64`
- `/c/PROGRA~2/MICROS~2/2019/Professional/Common7/IDE`
- `/c/PROGRA~2/WI3CF2~1/8.0/bin/x64`
- `/c/PROGRA~2/WI3CF2~1/8.0/bin/x86`
- `/c/PROGRA~2/MICROS~2/2019/Professional/TEAMTO~1/PERFOR~1/x64`
- `/c/PROGRA~2/MICROS~2/2019/Professional/TEAMTO~1/PERFOR~1`
- `/c/PROGRA~1/Java/jdk1.8.0_121/bin`
- `/c/usr/local/bin`
- `/c/usr/local/lib`

Finally, you need to create an alias `make` for `gnumake`.

For example, you might create a `.bashrc` file in your home directory containing this:

```
export CPP_SRC_HOME=$HOME/ml-cpp
VCVER=`/bin/ls -1 /c/PROGRA~2/MICROS~2/2019/Professional/VC/Tools/MSVC | tail -1`
VCBINDIR=/c/PROGRA~2/MICROS~2/2019/Professional/VC/Tools/MSVC/$VCVER/bin/Hostx64/x64:/c/PROGRA~2/MICROS~2/2019/Professional/Common7/IDE:/c/PROGRA~2/WI3CF2~1/8.0/bin/x64:/c/PROGRA~2/WI3CF2~1/8.0/bin/x86:/c/PROGRA~2/MICROS~2/2019/Professional/TEAMTO~1/PERFOR~1/x64:/c/PROGRA~2/MICROS~2/2019/Professional/TEAMTO~1/PERFOR~1
export JAVA_HOME=/c/PROGRA~1/Java/jdk1.8.0_121
export PATH="$CPP_SRC_HOME/build/distribution/platform/windows-x86_64/bin:$VCBINDIR:/mingw64/bin:$JAVA_HOME/bin:/c/usr/local/bin:/c/usr/local/lib:/bin:/c/Windows/System32:/c/Windows"
alias make=gnumake
```

### Operating system

64 bit Windows is required.

It is possible to build on Windows Server 2012r2, Windows Server 2016 and Windows 10.  Other versions may encounter problems.

### Windows 8 SDK

Download `sdksetup.exe` from https://developer.microsoft.com/en-us/windows/downloads/windows-8-sdk and run it.  Accept the default installation location, opt out of the customer experience improvement program and accept the license agreement.  Then install with all features selected.

### Microsoft Visual Studio 2019 Professional

_Make sure you install the Windows 8 SDK before this, as it cannot be installed afterwards._

The Professional edition requires an MSDN subscription. Download the installer, `vs_professional__904165217.1561988349.exe`, from <https://my.visualstudio.com/downloads>, and run it.

On the "Workloads" page that is displayed after a short while, check "Desktop development with C++".  Then click on the "Individual Components" tab and check "Windows Universal CRT SDK" (about half way down the list).  Then click "Install".

### Git for Windows

Download `Git-2.24.0.2-64-bit.exe` from <https://github.com/git-for-windows/git/releases/download/v2.24.0.windows.2>.

Install it using mainly the default options suggested by the installer, but on the feature selection dialog also check "On the desktop" in the "Additional icons" section.

Do not change the suggested default autocrlf setting (true), as Unix line endings do not work for unit test test files.

As well as providing a Git implementation, Git for Windows comes with Windows ports of a variety of common Unix commands, e.g. `grep`, `sed`, `awk`, `find`, and many others.

### GNU make

Download `make-4.2.1.tar.bz2` (or a more recent version) from <http://ftp.gnu.org/gnu/make/> . Extract it in a Git bash shell using the GNU tar that comes with Git for Windows, e.g.:

```
cd /c/tools
tar jxvf /z/cpp_src/make-4.2.1.tar.bz2
```

Start a command prompt using Start Menu -&gt; Apps -&gt; Visual Studio 2019 -&gt; x64 Native Tools Command Prompt for VS 2019, then in it type:

```
cd \tools\make-4.2.1
build_w32.bat
```

Finally, copy the file `gnumake.exe` from `C:\tools\make-4.2.1\WinRel` to `C:\usr\local\bin`.

### zlib

Whilst it is possible to download a pre-built version of `zlib1.dll`, for consistency we want one that links against the Visual Studio 2019 C runtime library. Therefore it is necessary to build zlib from source.

Download the source code from <http://zlib.net/> - the file is called `zlib1211.zip`. Unzip this file under `C:\tools`, so that you end up with a directory called `C:\tools\zlib-1.2.11`.

To build, start a command prompt using Start Menu -&gt; Apps -&gt; Visual Studio 2019 -&gt; x64 Native Tools Command Prompt for VS 2019, then in it type:

```
cd \tools\zlib-1.2.11
nmake -f win32/Makefile.msc AS=ml64 LOC="-D_WIN32_WINNT=0x0601 -DASMV -DASMINF -I." OBJA="inffasx64.obj gvmat64.obj inffas8664.obj"
nmake -f win32/Makefile.msc test
```

All the build output will end up in the top level `C:\tools\zlib-1.2.11` directory. Once the build is complete, copy `zlib1.dll` and `minigzip.exe` to `C:\usr\local\bin`. Copy `zlib.lib` and `zdll.lib` to `C:\usr\local\lib`. And copy `zlib.h` and `zconf.h` to `C:\usr\local\include`.

### libxml2

Download `libxml2-2.9.7.tar.bz2` from <ftp://xmlsoft.org/libxml2/> .

Extract it in a Git bash shell using the GNU tar that comes with Git for Windows, e.g.:

```
cd /c/tools
tar jxvf /z/cpp_src/libxml2-2.9.7.tar.bz2
```

Edit `C:\tools\libxml2-2.9.7\win32\Makefile.msvc` and change the following lines:

```
CFLAGS = $(CFLAGS) /D "NDEBUG" /O2
```

to:

```
CFLAGS = $(CFLAGS) /D "NDEBUG" /O2 /Zi /D_WIN32_WINNT=0x0601
```

(because we might as well generate a PDB file and we want to limit Windows API functions to those that exist in Windows Server 2008r2).

Start a command prompt using Start Menu -&gt; Apps -&gt; Visual Studio 2019 -&gt; x64 Native Tools Command Prompt for VS 2019, then in it type:

```
cd \tools\libxml2-2.9.7\win32
cscript configure.js iconv=no prefix=C:\usr\local
nmake
nmake install
```

### Boost 1.71.0

Download version 1.71.0 of Boost from <https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.bz2> . You must get this exact version, as the Machine Learning Makefiles expect it.

Assuming you chose the `.bz2` version, extract it in a Git bash shell using the GNU tar that comes with Git for Windows, e.g.:

```
cd /c/tools
tar jxvf /z/cpp_src/boost_1_71_0.tar.bz2
```

Edit `boost/unordered/detail/implementation.hpp` and change line 287 from:

```
    (17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \
```

to:

```
    (3ul)(17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \
```

Start a command prompt using Start Menu -&gt; Apps -&gt; Visual Studio 2019 -&gt; x64 Native Tools Command Prompt for VS 2019, then in it type:

```
cd \tools\boost_1_71_0
bootstrap.bat
b2 -j6 --layout=versioned --disable-icu --toolset=msvc-14.1 --build-type=complete -sZLIB_INCLUDE="C:\tools\zlib-1.2.11" -sZLIB_LIBPATH="C:\tools\zlib-1.2.11" -sZLIB_NAME=zdll --without-context --without-coroutine --without-graph_parallel --without-mpi --without-python architecture=x86 address-model=64 optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC define=_WIN32_WINNT=0x0601
b2 install --prefix=C:\usr\local --layout=versioned --disable-icu --toolset=msvc-14.1 --build-type=complete -sZLIB_INCLUDE="C:\tools\zlib-1.2.11" -sZLIB_LIBPATH="C:\tools\zlib-1.2.11" -sZLIB_NAME=zdll --without-context --without-coroutine --without-graph_parallel --without-mpi --without-python architecture=x86 address-model=64 optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC define=_WIN32_WINNT=0x0601
```

The Boost headers and appropriate libraries should end up in `C:\usr\local\include` and `C:\usr\local\lib` respectively.

### strptime

Visual Studio's C runtime library has a `strftime()` function but no `strptime()` function, and unfortunately it's needed for our date handling functionality.

There is an open source `strptime()` implementation from NetBSD. It needs some hacking about to get it to compile on Windows, but this is not an insurmountable challenge. Download `strptime.c` and `private.h` from <http://cvsweb.netbsd.org/bsdweb.cgi/src/lib/libc/time/strptime.c?rev=HEAD> and <http://cvsweb.netbsd.org/bsdweb.cgi/src/lib/libc/time/private.h?rev=HEAD> respectively. These files are also checked in in the `strptime` sub-directory of the directory containing these instructions, as are some patch files that modify the original files so that they'll compile on Windows.

In a Git bash shell, copy the original source files and patches to a temporary directory and apply the patches, e.g.:

```
cp -r strptime /c/tools
cd /c/tools/strptime
patch -i strptime.ucrt.patch
patch -i private.patch
```

Start a command prompt using Start Menu -&gt; Apps -&gt; Visual Studio 2019 -&gt; x64 Native Tools Command Prompt for VS 2019, then in it type:

```
cd C:\tools\strptime
cl -c -nologo -O2 -MD -DWIN32_LEAN_AND_MEAN -D_WIN32_WINNT=0x0601 -DNDEBUG strptime.c
lib -NOLOGO strptime.obj
copy strptime.lib C:\usr\local\lib
```

