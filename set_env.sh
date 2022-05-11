#!/bin/bash
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

# Set up a build environment, to ensure repeatable builds

umask 0002

# Modify some limits if not running in Docker (soft limits only, hence -S)
if [ ! -f /.dockerenv ]; then
    ulimit -S -c unlimited
    ulimit -S -n 1024
fi

# Set $CPP_SRC_HOME to be an absolute path to this script's location, as
# different builds will come from different repositories and go to different
# staging areas
MY_DIR=`dirname "$BASH_SOURCE"`
export CPP_SRC_HOME=`cd "$MY_DIR" && pwd`

# Platform
case `uname` in

    Darwin)
        SIMPLE_PLATFORM=macos
        BUNDLE_PLATFORM=darwin-`uname -m | sed 's/arm64/aarch64/'`
        export CMAKE_INSTALL_PREFIX="$CPP_SRC_HOME/build/distribution/platform/$BUNDLE_PLATFORM/controller.app/Contents/"
        export CMAKE_GENERATOR='Unix Makefiles'
        ;;

    Linux)
        SIMPLE_PLATFORM=linux
        if [ -z "$CPP_CROSS_COMPILE" ] ; then
            BUNDLE_PLATFORM=linux-`uname -m`
            export CMAKE_INSTALL_PREFIX="$CPP_SRC_HOME/build/distribution/platform/$BUNDLE_PLATFORM"
            export CMAKE_GENERATOR='Unix Makefiles'
        elif [ "$CPP_CROSS_COMPILE" = macosx ] ; then
            BUNDLE_PLATFORM=darwin-x86_64
            export CMAKE_GENERATOR='Unix Makefiles'
            export CMAKE_INSTALL_PREFIX="$CPP_SRC_HOME/build/distribution/platform/$BUNDLE_PLATFORM/controller.app/Contents/"
            export CMAKE_TOOLCHAIN_FILE="tc-darwin.cmake"
        else
            BUNDLE_PLATFORM=linux-$CPP_CROSS_COMPILE
            CMAKE_GENERATOR='Unix Makefiles'
            export CMAKE_INSTALL_PREFIX="$CPP_SRC_HOME/build/distribution/platform/$BUNDLE_PLATFORM"
            export CMAKE_TOOLCHAIN_FILE="tc-linux-aarch64.cmake"
        fi
        ;;

    MINGW*)
        SIMPLE_PLATFORM=windows
        BUNDLE_PLATFORM=windows-x86_64
        export CMAKE_GENERATOR='MinGW Makefiles'
        export LOCAL_DRIVE=${LOCAL_DRIVE:-"c"}
        export CMAKE_INSTALL_PREFIX="$CPP_SRC_HOME/build/distribution/platform/$BUNDLE_PLATFORM"
        ;;

    *)
        echo `uname 2>&1` "- unsupported operating system"
        exit 1
        ;;

esac

CPP_DISTRIBUTION_HOME="$CPP_SRC_HOME/build/distribution"
export CPP_PLATFORM_HOME="$CPP_DISTRIBUTION_HOME/platform/$BUNDLE_PLATFORM"

# Logical filesystem root
unset ROOT
if [ "$SIMPLE_PLATFORM" = "windows" ] ; then
    if [ -z "$LOCAL_DRIVE" ] ; then
        ROOT=/c
    else
        ROOT=/$LOCAL_DRIVE
    fi
    export ROOT
fi

# Compiler
unset COMPILER_PATH
if [ "$SIMPLE_PLATFORM" = "windows" ] ; then
    # We have to use 8.3 names for directories with spaces in the names, as some
    # tools won't quote paths with spaces correctly
    VCBASE=`(cd $ROOT && cygpath -m -s "Program Files (x86)/Microsoft Visual Studio/2019/Professional")`
    WINSDKBASE=`(cd $ROOT && cygpath -m -s "Program Files (x86)/Windows Kits")`
    PFX86_DIR=`cd $ROOT && cygpath -m -s "Program Files (x86)"`
    MSVC_DIR=`cd $ROOT/$PFX86_DIR && cygpath -m -s "Microsoft Visual Studio"`
    WIN_KITS_DIR=`cd $ROOT/$PFX86_DIR && cygpath -m -s "Windows Kits"`
    VCVER=`/bin/ls -1 $ROOT/$PFX86_DIR/$MSVC_DIR/2019/Professional/VC/Tools/MSVC | tail -1`
    UCRTVER=`/bin/ls -1 $ROOT/$WINSDKBASE/10/Include | tail -1`
    # NB: Some SDK tools are 32 bit only, hence the 64 bit SDK bin directory
    #     is followed by the 32 bit SDK bin directory
    COMPILER_PATH=$ROOT/$PFX86_DIR/$MSVC_DIR/2019/Professional/VC/Tools/MSVC/$VCVER/bin/Hostx64/x64:$ROOT/$PFX86_DIR/$MSVC_DIR/2019/Professional/Common7/IDE:$ROOT/$PFX86_DIR/$WIN_KITS_DIR/8.0/bin/x64:$ROOT/$PFX86_DIR/$WIN_KITS_DIR/8.0/bin/x86
    export INCLUDE=$ROOT/$VCBASE/VC/Tools/MSVC/$VCVER/atlmfc/include:$ROOT/$VCBASE/VC/Tools/MSVC/$VCVER/include:$ROOT/$WINSDKBASE/10/Include/$UCRTVER/ucrt:$ROOT/$WINSDKBASE/10/Include/$UCRTVER/shared:$ROOT/$WINSDKBASE/10/Include/$UCRTVER/um:$ROOT/$WINSDKBASE/10/Include/$UCRTVER/winrt:$ROOT/$WINSDKBASE/10/Include/$UCRTVER/cppwinrt
    export LIB=$ROOT/$VCBASE/VC/Tools/MSVC/$VCVER/atlmfc/lib/x64:$ROOT/$VCBASE/VC/Tools/MSVC/$VCVER/lib/x64:$ROOT/$WINSDKBASE/10/Lib/$UCRTVER/ucrt/x64:$ROOT/$WINSDKBASE/10/Lib/$UCRTVER/um/x64
    export LIBPATH=$PATH:$ROOT/$VCBASE/VC/Tools/MSVC/$VCVER/atlmfc/lib/x64:$ROOT/$VCBASE/VC/Tools/MSVC/$VCVER/lib/x64:$ROOT/$VCBASE/VC/Tools/MSVC/VCVER/lib/x86/store/REFERE~1:$ROOT/$WINSDKBASE/10/UNIONM~1/$UCRTVER:$ROOT/$WINSDKBASE/10/REFERE~1/$UCRTVER:$ROOT/Windows/MICROS~1.NET/FRAMEW~2/V40~1.303
fi

# Git
unset GIT_HOME
if [ -d "$ROOT/usr/local/git" ] ; then
    GIT_HOME=$ROOT/usr/local/git
fi

# Paths
case $SIMPLE_PLATFORM in

    linux)
        PATH=/usr/local/gcc103/bin:/usr/bin:/bin:/usr/local/gcc103/sbin:/usr/sbin:/sbin:/usr/local/bin:/usr/local/cmake/bin
        ;;

    macos)
        PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin
        ;;

    windows)
        PATH=/mingw64/bin:/usr/bin:/bin:$ROOT/Windows/System32:$ROOT/Windows:$ROOT/PROGRA~1/Python37/Scripts/
        PATH=$ROOT/usr/local/bin:$ROOT/usr/local/sbin:$PATH
        ;;

esac

if [ ! -z "$GIT_HOME" ] ; then
    PATH=$GIT_HOME/bin:$PATH
fi
if [ ! -z "$COMPILER_PATH" ] ; then
    PATH=$COMPILER_PATH:$PATH
fi

# Library paths
case $SIMPLE_PLATFORM in

    linux)
        export LD_LIBRARY_PATH=/usr/local/gcc103/lib64:/usr/local/gcc103/lib:/usr/lib:/lib
        ;;

    windows)
        # On Windows there's no separate library path - $PATH is used
        PATH=$CPP_PLATFORM_HOME/bin:$PATH:$ROOT/usr/local/lib
        ;;

esac

# do not cache in Jenkins
if [ -z "$JOB_NAME" ] ; then
    # check if CCACHE is available and use it if it's found
    if [ -d "/usr/lib/ccache" ] ; then
        PATH=/usr/lib/ccache/:$PATH
    # on Mac it should be /usr/local/lib when installed with brew
    elif [ -d "/usr/local/lib/ccache" ] ; then
        PATH=/usr/local/lib/ccache/:$PATH
    fi
fi


export PATH

# GNU make
case $SIMPLE_PLATFORM in

    linux*)
        MAKE=`which make`
        ;;

    macos|windows)
        MAKE=`which gnumake`
        ;;

esac

export MAKE

# Localisation - don't use any locale specific functionality
export LC_ALL=C

# Unset environment variables that affect the choice of compiler/linker, or
# where the compiler/linker look for dependencies - we only want what's in our
# Makefiles
unset CPP
unset CC
unset CXX
unset LD
unset CPPFLAGS
unset CFLAGS
unset CXXFLAGS
unset LDFLAGS
if [ "$SIMPLE_PLATFORM" != "windows" ] ; then
    unset CPATH
    unset C_INCLUDE_PATH
    unset CPLUS_INCLUDE_PATH
    unset LIBRARY_PATH
fi

# Tell the build to keep going under Jenkins so that we get as many errors as
# possible from an automated build (Jenkins sets $JOB_NAME)
if [ -n "$JOB_NAME" ] ; then
    export ML_KEEP_GOING=1
fi

