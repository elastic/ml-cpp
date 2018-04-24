# Machine Learning Build Machine Setup for Windows (simple)

These instructions are an alternative to the "from scratch" instructions in [windows.md](build-setup/windows.md).

This setup relies on pre-built 3rd party components previously built by the "from scratch" instructions and uploaded to s3. However, you get a fully working dev environment for ml-cpp.

### Operating system

64 bit Windows is required.

It is possible to build on Windows Server 2012r2, Windows Server 2016 and Windows 10.  Other versions may encounter problems.

### Base requirements

Follow the instructions (**only!**) for Windows 8 SDK and Git for Windows from [windows.md](build-setup/windows.md).

### Microsoft C++ build tools

Download "Build Tools for Visual Studio 2017" from https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017 and run it.

### Clone ml-cpp

The github repository contains a script to download the remaining dependencies, therefore it is best to clone ml-cpp to your preferred location. Afterwards execute the powershell script from: `dev-tools\download_windows_deps.ps1`.

### Environment

Create a `.bashrc` similar to that recommended in [windows.md](build-setup/windows.md), but with a couple of extra environment variables.  The build system lets you define `VCBASE` and `WINSDKBASE` to point it to the right compiler and SDK. In addition you have to put the directories into PATH as well. This is how it could look:

```
export VCBASE=PROGRA~2/MICROS~2/2017/BUILDT~1
export WINSDKBASE=PROGRA~2/WI3CF2~1

export CPP_SRC_HOME=$HOME/ml-cpp
VCVER=`/bin/ls -1 /c/$VCBASE/VC/Tools/MSVC | tail -1`
VCBINDIR=/c/$VCBASE/VC/Tools/MSVC/$VCVER/bin/HostX64/x64:/c/$VCBASE/Common7/IDE:/c/$WINSDKBASE/8.0/bin/x64:/c/$WINSDKBASE/8.0/bin/x86:/c/$VCBASE/TEAMTO~1/PERFOR~1/x64:/c/$VCBASE/TEAMTO~1/PERFOR~1
export JAVA_HOME=/c/PROGRA~1/Java/jdk1.8.0_121
export PATH="$CPP_SRC_HOME/build/distribution/platform/windows-x86_64/bin:$VCBINDIR:/mingw64/bin:$JAVA_HOME/bin:/c/usr/local/bin:/c/usr/local/lib:/bin:/c/Windows/System32:/c/Windows"
alias make=gnumake
```

