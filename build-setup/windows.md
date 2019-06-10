# Machine Learning Build Machine Setup for Windows

These instructions are for setting up a Windows build machine from scratch.

For subsequent machines, the setup can be simplified by copying components from the first machine - see [windows_simple.md](windows_simple.md).

The build system assumes Windows itself is installed on the `C:` drive. If this is not the case on your machine, the Makefiles will not work.

On Windows, the build system runs within Git for Windows. This provides a bash shell that enables our Unix build system to work on Windows without alteration.

If you use Gradle to build Machine Learning you do not need to work in Git Bash, as Gradle tasks will run it when necessary.  In order to develop interactively on Windows, you must work in Git Bash.  You will need the following environment variables to be defined:

- `CPP_SRC_HOME` - This must contain the absolute path of the top level source code directory.

You will also need to add several directories to your `PATH` environment variable. These need to be in the MinGW format, i.e. beginning with /drive letter/ followed by the directory path using forward slashes instead of backslashes, and with no spaces in the directory names. Where a Windows directory name contains a space, you must use the 8.3 version of that directory name (which you can find using `dir /x` at a command prompt). The directories that need to be added are:

- `/c/PROGRA~2/MICROS~2/2017/Professional/VC/Tools/MSVC/14.11.25503/bin/HostX64/x64`
- `/c/PROGRA~2/MICROS~2/2017/Professional/Common7/IDE`
- `/c/PROGRA~2/WINDOW~4/8.0/bin/x64`
- `/c/PROGRA~2/WINDOW~4/8.0/bin/x86`
- `/c/PROGRA~2/MICROS~2/2017/Professional/TEAMTO~1/PERFOR~1/x64`
- `/c/PROGRA~2/MICROS~2/2017/Professional/TEAMTO~1/PERFOR~1`
- `/c/PROGRA~1/Java/jdk1.8.0_121/bin`
- `/c/usr/local/bin`
- `/c/usr/local/lib`

Finally, you need to create an alias `make` for `gnumake`.

For example, you might create a `.bashrc` file in your home directory containing this:

```
export CPP_SRC_HOME=$HOME/ml-cpp
VCVER=`/bin/ls -1 /c/PROGRA~2/MICROS~2/2017/Professional/VC/Tools/MSVC | tail -1`
VCBINDIR=/c/PROGRA~2/MICROS~2/2017/Professional/VC/Tools/MSVC/$VCVER/bin/HostX64/x64:/c/PROGRA~2/MICROS~2/2017/Professional/Common7/IDE:/c/PROGRA~2/WINDOW~4/8.0/bin/x64:/c/PROGRA~2/WINDOW~4/8.0/bin/x86:/c/PROGRA~2/MICROS~2/2017/Professional/TEAMTO~1/PERFOR~1/x64:/c/PROGRA~2/MICROS~2/2017/Professional/TEAMTO~1/PERFOR~1
export JAVA_HOME=/c/PROGRA~1/Java/jdk1.8.0_121
export PATH="$CPP_SRC_HOME/build/distribution/platform/windows-x86_64/bin:$VCBINDIR:/mingw64/bin:$JAVA_HOME/bin:/c/usr/local/bin:/c/usr/local/lib:/bin:/c/Windows/System32:/c/Windows"
alias make=gnumake
```

### Operating system

64 bit Windows is required.

It is possible to build on Windows Server 2012r2, Windows Server 2016 and Windows 10.  Other versions may encounter problems.

### Windows 8 SDK

Download `sdksetup.exe` from https://developer.microsoft.com/en-us/windows/downloads/windows-8-sdk and run it.  Accept the default installation location, opt out of the customer experience improvement program and accept the license agreement.  Then install with all features selected.

### Microsoft Visual Studio 2017 Professional

_Make sure you install the Windows 8 SDK before this, as it cannot be installed afterwards._

The Professional edition requires an MSDN subscription. Download the installer, `vs_Professional.exe`, from <https://my.visualstudio.com/downloads>, and run it.

On the "Workloads" page that is displayed after a short while, check "Desktop development with C++".  Then click on the "Individual Components" tab and check "Windows Universal CRT SDK" (about half way down the list).  Then click "Install".

### Git for Windows

Download `Git-2.14.1-64-bit.exe` from <https://github.com/git-for-windows/git/releases/tag/v2.14.1.windows.1>.

Install it using mainly the default options suggested by the installer, but on the feature selection dialog also check "On the desktop" in the "Additional icons" section.

Do not change the suggested default autocrlf setting (true), as Unix line endings do not work for unit test test files.

As well as providing a Git implementation, Git for Windows comes with Windows ports of a variety of common Unix commands, e.g. `grep`, `sed`, `awk`, `find`, and many others.

### GNU make

Download `make-4.2.1.tar.bz2` (or a more recent version) from <http://ftp.gnu.org/gnu/make/> . Extract it in a Git bash shell using the GNU tar that comes with Git for Windows, e.g.:

```
cd /c/tools
tar jxvf /z/cpp_src/make-4.2.1.tar.bz2
```

Start a command prompt using Start Menu -&gt; Apps -&gt; Visual Studio 2017 -&gt; x64 Native Tools Command Prompt for VS 2017, then in it type:

```
cd \tools\make-4.2.1
build_w32.bat
```

Finally, copy the file `gnumake.exe` from `C:\tools\make-4.2.1\WinRel` to `C:\usr\local\bin`.

### zlib

Whilst it is possible to download a pre-built version of `zlib1.dll`, for consistency we want one that links against the Visual Studio 2017 C runtime library. Therefore it is necessary to build zlib from source.

Download the source code from <http://zlib.net/> - the file is called `zlib1211.zip`. Unzip this file under `C:\tools`, so that you end up with a directory called `C:\tools\zlib-1.2.11`.

To build, start a command prompt using Start Menu -&gt; Apps -&gt; Visual Studio 2017 -&gt; x64 Native Tools Command Prompt for VS 2017, then in it type:

```
cd \tools\zlib-1.2.11
nmake -f win32/Makefile.msc AS=ml64 LOC="-D_WIN32_WINNT=0x0601 -DASMV -DASMINF -I." OBJA="inffasx64.obj gvmat64.obj inffas8664.obj"
nmake -f win32/Makefile.msc test
```

All the build output will end up in the top level `C:\tools\zlib-1.2.11` directory. Once the build is complete, copy `zlib1.dll` and `minigzip.exe` to `C:\usr\local\bin`. Copy `zlib.lib` and `zdll.lib` to `C:\usr\local\lib`. And copy `zlib.h` and `zconf.h` to `C:\usr\local\include`.

### cppunit

Download the latest version of cppunit from <http://dev-www.libreoffice.org/src/cppunit-1.13.2.tar.gz> (or if that no longer exists by the time you read this, find the relevant link on <http://dev-www.libreoffice.org/src>).

The file will be something along the lines of `cppunit-1.13.2.tar.bz2`. Extract it in a Git bash shell using the GNU tar that comes with Git for Windows, e.g.:

```
cd /c/tools
tar jxvf /z/cpp_src/cppunit-1.13.2.tar.bz2
```

Double click `C:\tools\cppunit-1.13.2\src\CppUnitLibraries2010.sln`. Visual Studio 2017 should load. Skip the offer to install ATL and MFC (as we don't care about the test program, which is the only thing that requires them).

Right click on the "Solution 'CppUnitLibraries2010' (3 projects)" on the right hand side of the screen and choose "Retarget solution...".  Accept the defaults in the resulting dialog box and click "OK".

Select "Release" in the "Solution Configurations" dropdown and "x64" in the "Solution Platforms" dropdown. Then press Ctrl + Shift + B to build. The IDE will attempt to build all five projects - some of them will fail to build, but that doesn't matter as the two important ones are the static and dynamic cppunit libraries.

Once the builds have completed, go to `C:\tools\cppunit-1.13.2\src\cppunit\ReleaseDll` in Windows Explorer, and copy `cppunit_dll.dll` and `cppunit_dll.lib` to `C:\usr\local\lib`. Then go to `C:\tools\cppunit-1.13.2\src\cppunit\Release` and copy `cppunit.lib` to `C:\usr\local\lib`.

Finally, the easiest way to copy all the headers whilst excluding other non-headers in the same directories is using a Git bash shell:

```
cd /c/tools/cppunit-1.13.2/include
tar cf - `find cppunit -name '*.h'` | (cd /c/usr/local/include && tar xvf -)
```

### libxml2

Download `libxml2-2.9.4.tar.bz2` from <ftp://xmlsoft.org/libxml2/> .

Extract it in a Git bash shell using the GNU tar that comes with Git for Windows, e.g.:

```
cd /c/tools
tar jxvf /z/cpp_src/libxml2-2.9.4.tar.bz2
```

Edit `C:\tools\libxml2-2.9.4\win32\Makefile.msvc` and change the following lines:

```
CFLAGS = $(CFLAGS) /D "NDEBUG" /O2
```

to:

```
CFLAGS = $(CFLAGS) /D "NDEBUG" /O2 /Zi /D_WIN32_WINNT=0x0601
```

(because we might as well generate a PDB file and we want to limit Windows API functions to those that exist in Windows Server 2008r2).

Start a command prompt using Start Menu -&gt; Apps -&gt; Visual Studio 2017 -&gt; x64 Native Tools Command Prompt for VS 2017, then in it type:

```
cd \tools\libxml2-2.9.4\win32
cscript configure.js iconv=no prefix=C:\usr\local
nmake
nmake install
```

### expat

Download expat from <https://github.com/libexpat/libexpat/releases/download/R_2_2_6/expat-2.2.6.tar.bz2>.

Extract it in a Git bash shell using the GNU tar that comes with Git for Windows, e.g.:

```
cd /c/tools
tar jxvf /z/cpp_src/expat-2.2.6.tar.bz2
```

Double click `C:\tools\expat-2.2.6\expat.sln`. Visual Studio 2017 should load.

Select "Release" in the "Solution Configurations" dropdown and "x64" in the "Solution Platforms" dropdown.

Right click on "expat_static" on the right hand side of the screen and choose "Properties...".  Then in the dialog box, go to Configuration Properties -&gt; C/C++ -&gt; Code Generation and change "Runtime Library" to "Multi-threaded DLL (/MD)".  Also go to Configuration Properties -&gt; Librarian -&gt; General and change "Output File" from "..\win32\bin\Release\libexpatMT.lib" to "..\x64\bin\Release\libexpatMD.lib".  Then click "OK".

Build the static library by right clicking "expat_static" again and choosing "Build".

Then from `C:\tools\expat-2.2.6\x64\bin\Release` manually copy `libexpatMD.lib` into `C:\usr\local\lib`, and from `C:\tools\expat-2.2.6\lib` manually copy `expat.h` and `expat_external.h` into `C:\usr\local\include`.

### APR

Download the Windows source for version 1.7.0 of the Apache Portable Runtime (APR) from <http://archive.apache.org/dist/apr/apr-1.7.0-win32-src.zip>.

Unzip the file into `C:\tools`. You should end up with a subdirectory called `apr-1.7.0`. Rename this to simply `apr`.

Edit `C:\tools\apr\include\apr.hw` and change lines 87-89 from:

```
/* Restrict the server to a subset of Windows XP header files by default
 */
#define _WIN32_WINNT 0x0501
```

to:

```
/* Restrict the server to a subset of Windows 7 header files by default
 */
#define _WIN32_WINNT 0x0601
```

Start a command prompt using Start Menu -&gt; Apps -&gt; Visual Studio 2017 -&gt; x64 Native Tools Command Prompt for VS 2017, then in it type:

```
cd C:\tools\apr
copy include\apr.hw include\apr.h
copy include\apr.hw apr.h
nmake -f Makefile.win ARCH="x64 Release" buildall
nmake -f Makefile.win PREFIX=C:\usr\local ARCH="x64 Release" install
```

### APR iconv

Download the Windows source for version 1.2.2 of the Apache Portable Runtime (APR) version of iconv from <http://archive.apache.org/dist/apr/apr-iconv-1.2.2-win32-src.zip>.

Unzip the file into `C:\tools`. You should end up with a subdirectory called `apr-iconv-1.2.2`. Rename this to simply `apr-iconv`.

Note that this does not install the iconv utilities - this will be done when you build the APR utility library, which you must do next.

### APR util

Download the Windows source for version 1.6.1 of the Apache Portable Runtime (APR) utilities from <http://archive.apache.org/dist/apr/apr-util-1.6.1-win32-src.zip>.

Unzip the file into `C:\tools`. You should end up with a subdirectory called `apr-util-1.6.1`. Rename this to simply `apr-util`.

Right-click on the directory `C:\tools\apr-util\xml`, go to Properties -&gt; Security -&gt; CREATOR OWNER -&gt; Edit..., check Full Control -&gt; Allow and then OK all the open dialog boxes.

In both `C:\tools\apr-util\apr-util.mak` and `C:\tools\apr-util\libaprutil.mak` globally replace `./xml/expat/lib` with `../expat-2.2.6/lib`.

Start a command prompt using Start Menu -&gt; Apps -&gt; Visual Studio 2017 -&gt; x64 Native Tools Command Prompt for VS 2017, then in it type:

```
cd C:\tools\apr-util
copy include\apr_ldap.hw include\apr-ldap.h
copy include\apu.hw include\apu.h
copy include\apu_want.hw include\apr_want.h
nmake -f Makefile.win XML_PARSER=..\expat-2.2.6\x64\bin\Release\libexpatMD ARCH="x64 Release" buildall
nmake -f Makefile.win XML_PARSER=..\expat-2.2.6\x64\bin\Release\libexpatMD PREFIX=C:\usr\local ARCH="x64 Release" install
```

It is extremely important with this library that it is NOT linked to all sorts of unnecessary 3rd party libraries that we don't need (some of which are GPL). Therefore, once the build is complete, use `dumpbin` to confirm that `libaprutil-1.dll` does NOT have dependencies like MySQL, PostgreSQL, Berkeley DB, GNU DB Manager, LDAP, SQLite or ODBC. If it does, do the build again having first researched why the extra databases were linked.

### log4cxx

APR must be downloaded, the source put in the directories specified in the three previous sections, and successfully built before you start to build log4cxx.

Download `apache-log4cxx-0.10.0.tar.gz` from one of the mirrors listed at <http://www.apache.org/dyn/closer.cgi/logging/log4cxx/0.10.0/apache-log4cxx-0.10.0.tar.gz> .

Unzip the file into `C:\tools`. You should end up with a subdirectory called `apache-log4cxx-0.10.0`.

Edit the file `C:\tools\apache-log4cxx-0.10.0\src\main\include\log4cxx\log4cxx.hw`, and change:

```
#define LOG4CXX_LOGCHAR_IS_UTF8 0
```

to:

```
#define LOG4CXX_LOGCHAR_IS_UTF8 1
```

and, in the same file, delete both:

```
template class LOG4CXX_EXPORT std::allocator<T>; \
template class LOG4CXX_EXPORT std::vector<T>; \
```

and:

```
extern template class LOG4CXX_EXPORT std::allocator<T>; \
extern template class LOG4CXX_EXPORT std::vector<T>; \
```

Edit the file `C:\tools\apache-log4cxx-0.10.0\src\main\include\log4cxx\private\log4cxx\private.hw`, and change:

```
#define LOG4CXX_CHARSET_UTF8 0
```

to:

```
#define LOG4CXX_CHARSET_UTF8 1
```

and, in the same file, change:

```
#define LOG4CXX_FORCE_WIDE_CONSOLE 1
```

to:

```
#define LOG4CXX_FORCE_WIDE_CONSOLE 0
```

Also, in `C:\tools\apache-log4cxx-0.10.0\src\main\cpp\stringhelper.cpp`, add:

```
#include <iterator>
```

immediately after the other `#include`s.

Then run `C:\tools\apache-log4cxx-0.10.0\configure.bat`.

Next, double click on `C:\tools\apache-log4cxx-0.10.0\projects\log4cxx.dsw` to start the Visual Studio 2017 IDE. When asked if you want to do a one-way upgrade, click OK.

Once conversion is complete, select "Release" in the "Solution Configurations" dropdown. Then select "Configuration Manager..." from the "Solution Platforms" dropdown, and in the dialog box that appears, select "x64" in the "Active solution platform:" dropdown. Unfortunately this isn't available for log4cxx itself, so then select "&lt;New...&gt;" in the "Platform" column for log4cxx. In the next dialog box, select "x64" in the "Type or select the new platform:" dropdown, and uncheck the "Create new solution platforms" option. Then press "OK" and "Close" on the two dialog boxes to save the changes.

In property manager tree (on the right hand side), click on log4cxx, then click the properties button. In the resulting dialog box, go to Configuration Properties -&gt; Linker -&gt; General, select "Additional Library Directories", choose "&lt;Edit...&gt;" from the dropdown and add:

```
C:\usr\local\lib
```

Then click OK. Then in the previous dialog box, go to Configuration Properties -&gt; Linker -&gt; Input, select "Additional Dependencies", choose "&lt;Edit...&gt;" from the dropdown and add:

```
libapr-1.lib
libaprutil-1.lib
```

Then click OK, and OK again on the previous dialog box too.

Right click on the "Solution 'log4cxx' (4 projects)" on the right hand side of the screen and choose "Retarget solution...".  Accept the defaults in the resulting dialog box and click "OK".

Press Ctrl+Shift+B to build the solution, and hopefully log4cxx will build. The other three projects will NOT build, but that doesn't matter as they were already built in the preceeding steps. Assuming log4cxx itself builds, you can quit the IDE.

Then from `C:\tools\apache-log4cxx-0.10.0\projects\Release`, manually copy:

- `log4cxx.dll` into `C:\usr\local\bin`
- `log4cxx.lib` into `C:\usr\local\lib`

Finally, the easiest way to copy all the headers whilst excluding other non-headers in the same directories is using a Git bash shell:

```
cd /c/tools/apache-log4cxx-0.10.0/src/main/include
tar cf - `find log4cxx -name '*.h'` | (cd /c/usr/local/include && tar xvf -)
```

### Boost 1.65.1

Download version 1.65.1 of Boost from <http://sourceforge.net/projects/boost/files/boost/1.65.1/> . You must get this exact version, as the Machine Learning Makefiles expect it.

Assuming you chose the `.bz2` version, extract it in a Git bash shell using the GNU tar that comes with Git for Windows, e.g.:

```
cd /c/tools
tar jxvf /z/cpp_src/boost_1_65_1.tar.bz2
```

Edit `boost/unordered/detail/implementation.hpp` and change line 270 from:

```
    (17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \
```

to:

```
    (3ul)(17ul)(29ul)(37ul)(53ul)(67ul)(79ul) \
```

Start a command prompt using Start Menu -&gt; Apps -&gt; Visual Studio 2017 -&gt; x64 Native Tools Command Prompt for VS 2017, then in it type:

```
cd \tools\boost_1_65_1
bootstrap.bat
b2 -j6 --layout=versioned --disable-icu --toolset=msvc-14.1 --build-type=complete -sZLIB_INCLUDE="C:\tools\zlib-1.2.11" -sZLIB_LIBPATH="C:\tools\zlib-1.2.11" -sZLIB_NAME=zdll --without-context --without-coroutine --without-graph_parallel --without-log --without-mpi --without-python architecture=x86 address-model=64 optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=_WIN32_WINNT=0x0601
b2 install --prefix=C:\usr\local --layout=versioned --disable-icu --toolset=msvc-14.1 --build-type=complete -sZLIB_INCLUDE="C:\tools\zlib-1.2.11" -sZLIB_LIBPATH="C:\tools\zlib-1.2.11" -sZLIB_NAME=zdll --without-context --without-coroutine --without-graph_parallel --without-log --without-mpi --without-python architecture=x86 address-model=64 optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=_WIN32_WINNT=0x0601
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

Start a command prompt using Start Menu -&gt; Apps -&gt; Visual Studio 2017 -&gt; x64 Native Tools Command Prompt for VS 2017, then in it type:

```
cd C:\tools\strptime
cl -c -nologo -O2 -MD -DWIN32_LEAN_AND_MEAN -D_WIN32_WINNT=0x0601 -DNDEBUG strptime.c
lib -NOLOGO strptime.obj
copy strptime.lib C:\usr\local\lib
```

