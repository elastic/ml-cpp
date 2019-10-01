#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Set up a build environment, to ensure repeatable builds

umask 0002

# Modify some limits (soft limits only, hence -S)
ulimit -S -c unlimited
ulimit -S -n 1024

# Set $CPP_SRC_HOME to be an absolute path to this script's location, as
# different builds will come from different repositories and go to different
# staging areas
MY_DIR=`dirname "$BASH_SOURCE"`
export CPP_SRC_HOME=`cd "$MY_DIR" && pwd`

# Platform
case `uname` in

    Darwin)
        SIMPLE_PLATFORM=macos
        BUNDLE_PLATFORM=darwin-x86_64
        ;;

    Linux)
        SIMPLE_PLATFORM=linux
        if [ -z "$CPP_CROSS_COMPILE" ] ; then
            BUNDLE_PLATFORM=linux-x86_64
        elif [ "$CPP_CROSS_COMPILE" = macosx ] ; then
            BUNDLE_PLATFORM=darwin-x86_64
        else
            echo "Cannot cross compile to $CPP_CROSS_COMPILE"
            exit 1
        fi
        ;;

    MINGW*)
        SIMPLE_PLATFORM=windows
        BUNDLE_PLATFORM=windows-x86_64
        ;;

    *)
        echo `uname 2>&1` "- unsupported operating system"
        exit 2
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
    PFX86_DIR=`cd $ROOT && cygpath -m -s "Program Files (x86)"`
    MSVC_DIR=`cd $ROOT/$PFX86_DIR && cygpath -m -s "Microsoft Visual Studio 12.0"`
    WIN_KITS_DIR=`cd $ROOT/$PFX86_DIR && cygpath -m -s "Windows Kits"`
    # NB: Some SDK tools are 32 bit only, hence the 64 bit SDK bin directory
    #     is followed by the 32 bit SDK bin directory
    COMPILER_PATH=$ROOT/$PFX86_DIR/$MSVC_DIR/VC/bin/x86_amd64:$ROOT/$PFX86_DIR/$MSVC_DIR/VC/bin:/c/$PFX86_DIR/$MSVC_DIR/Common7/IDE:/c/$PFX86_DIR/$WIN_KITS_DIR/8.1/bin/x64:/c/$PFX86_DIR/$WIN_KITS_DIR/8.1/bin/x86
fi

# Git
unset GIT_HOME
if [ -d "$ROOT/usr/local/git" ] ; then
    GIT_HOME=$ROOT/usr/local/git
fi

# Paths
case $SIMPLE_PLATFORM in

    linux)
        PATH=/usr/local/gcc62/bin:/usr/bin:/bin:/usr/local/gcc62/sbin:/usr/sbin:/sbin
        ;;

    linux-musl)
        PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin
        ;;

    macos)
        PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin
        ;;

    windows)
        PATH=/mingw64/bin:/usr/bin:/bin:$ROOT/Windows/System32:$ROOT/Windows
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
        export LD_LIBRARY_PATH=/usr/local/gcc62/lib64:/usr/local/gcc62/lib:/usr/lib:/lib
        ;;

    linux-musl)
        export LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib:/usr/lib:/lib
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
if [ "$SIMPLE_PLATFORM" = "windows" ] ; then
    unset INCLUDE
    unset LIBPATH
else
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

# Finally, switch off debug if it's currently switched on
unset ML_DEBUG

