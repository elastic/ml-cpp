#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#

# This script builds and installs the required 3rd-party dependencies for building ml-cpp on Windows.
# It presumes a working compilation environment, e.g. VisualStudio 2022 with the latest Windows 10 SDK, and GitBash installed.
# The script itself uses nothing more complex than native PowerShell commands (and a few lines of Bash script), it explicitly
# does not make use of a package manager, such as Chocolately (choco). 

# Set up the compilation environment
& 'C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64 -HostArch amd64

# Determine the path to the ml-cpp repo
cd $PSScriptRoot
$ml_cpp_root = git rev-parse --show-toplevel

# Create tools directory
mkdir c:\tools

# Create installation directories
mkdir C:\usr\local\bin -ea 0
mkdir C:\usr\local\lib -ea 0
mkdir C:\usr\local\include -ea 0

# Build and install zlib 1.3.1
Write-Host "--- Installing zlib 1.3.1"
$Destination="C:\tools"
$Archive="zlib131.zip"
$ZipSource="http://zlib.net/$Archive"
$ZipDestination="\tools\$Archive"
(New-Object Net.WebClient).DownloadFile($ZipSource, $ZipDestination)
Add-Type -assembly "system.io.compression.filesystem"
[IO.Compression.ZipFile]::ExtractToDirectory($ZipDestination, $Destination)
cd \tools\zlib-1.3.1
nmake -f win32/Makefile.msc LOC="-D_WIN32_WINNT=0x0601"
nmake -f win32/Makefile.msc test
Copy-Item zlib1.dll -Destination \usr\local\bin
Copy-Item .\minigzip.exe -Destination \usr\local\bin
Copy-Item .\zlib.lib -Destination \usr\local\lib
Copy-Item .\zdll.lib -Destination \usr\local\lib
Copy-Item .\zlib.h  -Destination C:\usr\local\include\
Copy-Item .\zconf.h  -Destination C:\usr\local\include\
Write-Host "--- Done Installing zlib 1.3.1"

# Build and install libxml2 2.9.14
Write-Host "--- Installing libxml2 2.9.14"
cd c:\tools
$Archive="libxml2-2.9.14.tar.xz"
$ZipSource="https://download.gnome.org/sources/libxml2/2.9/$Archive"
$ZipDestination="\tools\$Archive"
(New-Object Net.WebClient).DownloadFile($ZipSource, $ZipDestination)
C:\Progra~1\git\bin\bash.exe -c "tar Jxvf libxml2-2.9.14.tar.xz"
cd \tools\libxml2-2.9.14\win32
(Get-Content .\Makefile.msvc) -replace """NDEBUG"" /O2" , """NDEBUG"" /O2 /Zi /D_WIN32_WINNT=0x0601" | Set-Content .\Makefile.msvc
cscript configure.js iconv=no prefix=C:\usr\local
nmake
nmake install
Write-Host "--- Done Installing libxml2 2.9.14"

# Build and install Boost 1.86.0
Write-Host "--- Installing boost 1.86.0"
cd c:\tools
$Archive="boost_1_86_0.tar.bz2"
$ZipSource="https://boostorg.jfrog.io/artifactory/main/release/1.86.0/source/$Archive"
$ZipDestination="\tools\$Archive"
(New-Object Net.WebClient).DownloadFile($ZipSource, $ZipDestination)
C:\Progra~1\git\bin\bash.exe -c "tar jxvf boost_1_86_0.tar.bz2"
cd \tools\boost_1_86_0
(Get-Content .\boost\unordered\detail\prime_fmod.hpp) -replace "{13ul" , "{3ul, 13ul" | Set-Content .\boost\unordered\detail\prime_fmod.hpp
c:\tools\boost_1_86_0\bootstrap.bat
c:\tools\boost_1_86_0\b2 -j6 install --prefix=C:\usr\local --layout=versioned --disable-icu --toolset=msvc-14.4 cxxflags="-std:c++17" linkflags="-std:c++17" --build-type=complete -sZLIB_INCLUDE="C:\tools\zlib-1.3.1" -sZLIB_LIBPATH="C:\tools\zlib-1.3.1" -sZLIB_NAME=zdll --without-context --without-coroutine --without-graph_parallel --without-mpi --without-python architecture=x86 address-model=64 optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC define=_WIN32_WINNT=0x0601
Write-Host "--- Done Installing boost 1.86.0"


# Build and install strptime
Write-Host "--- Installing strptime"
cd c:\tools
cp $ml_cpp_root\build-setup\strptime\ strptime -Recurse
cd .\strptime\
patch -i strptime.ucrt.patch
patch -i private.patch
cl -c -nologo -O2 -MD -DWIN32_LEAN_AND_MEAN -D_WIN32_WINNT=0x0601 -DNDEBUG strptime.c
lib -NOLOGO strptime.obj

copy strptime.lib C:\usr\local\lib
Write-Host "--- Done Installing strptime"

# Download and install python 3.10
Write-Host "--- Installing python 3.10.10"
cd c:\tools
$PythonArchive="python-3.10.10-amd64.exe"
$ZipSource="https://www.python.org/ftp/python/3.10.10/$PythonArchive"
$ZipDestination="\tools\$PythonArchive"
(New-Object Net.WebClient).DownloadFile($ZipSource, $ZipDestination)
Start-Process -FilePath  C:\tools\$PythonArchive -ArgumentList "TargetDir=C:\Python310 InstallAllUsers=1 PrependPath=1 Include_pip=1 /quiet /passive" -NoNewWindow -Wait
$env:PATH = "C:\Python310\Scripts\;C:\Python310\;" + $env:PATH
Write-Host "--- Done Installing python 3.10.10"

# Install python modules required by PyTorch build
Write-Host "--- Installing python modules"
pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
Write-Host "--- Done Installing python modules"

# Move the experimental MS libomp libraries out of the way temporarily, as we don't want to inadvertently redistribute them.
$f = ls 'C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\' -name | Select-Object -Last 1
mv "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\$f\lib\x64\libomp.lib" "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\$f\lib\x64\libomp.lib.bak"
mv "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\$f\lib\x64\libompd.lib" "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\$f\lib\x64\libompd.lib.bak"


# Build and install PyTorch (this may take a while)
Write-Host "--- Installing PyTorch locally"
cd c:\tools
git clone --depth=1 --branch=v2.5.1 https://github.com/pytorch/pytorch.git
cd pytorch
git submodule sync
git submodule update --init --recursive
(Get-Content ".\torch\csrc\jit\codegen\fuser\cpu\fused_kernel.cpp" -Raw) -replace "(?ms)std::unique_ptr.*?pipe.*?;" , "std::unique_ptr<FILE> pipe;" -replace "(?ms)intptr_t r = _wspawnve.*?return r;" , "return -1;" | Set-Content -Path ".\torch\csrc\jit\codegen\fuser\cpu\fused_kernel.cpp"
set BUILD_TEST=OFF
set BUILD_CAFFE2=OFF
set USE_NUMPY=OFF
set USE_DISTRIBUTED=OFF
set USE_MKLDNN=ON
set USE_QNNPACK=OFF
set USE_PYTORCH_QNNPACK=OFF
set USE_XNNPACK=OFF
set MSVC_Z7_OVERRIDE=OFF
set PYTORCH_BUILD_VERSION=2.5.1
set PYTORCH_BUILD_NUMBER=1
python setup.py install
Write-Host "--- Done Installing PyTorch locally"


Write-Host "--- Installing PyTorch globally"
cd c:\tools\pytorch
rm c:\usr\local\include\pytorch -recurse -ea 0
mkdir c:\usr\local\include\pytorch
cp torch\include\* c:\usr\local\include\pytorch\ -Recurse
cp torch\lib\torch_cpu.dll c:\usr\local\bin\
cp torch\lib\torch_cpu.lib c:\usr\local\lib\
cp torch\lib\c10.dll c:\usr\local\bin\
cp torch\lib\c10.lib c:\usr\local\lib\
cp torch\lib\fbgemm.dll c:\usr\local\bin\
cp torch\lib\fbgemm.lib c:\usr\local\lib\
cp torch\lib\asmjit.dll c:\usr\local\bin\
cp torch\lib\asmjit.lib c:\usr\local\lib\
Write-Host "--- Done Installing PyTorch globally"

# Move the libomp libraries back where we found them
mv "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\$f\lib\x64\libomp.lib.bak" "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\$f\lib\x64\libomp.lib"
mv "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\$f\lib\x64\libompd.lib.bak" "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\$f\lib\x64\libompd.lib"

# Uninstall python, it was only needed for the PyTorch build
& c:\tools\$PythonArchive /uninstall /quiet
rm c:\Python310 -recurse -force

# Remove c:\tools entirely
cd c:\
rm c:\tools -recurse -force

