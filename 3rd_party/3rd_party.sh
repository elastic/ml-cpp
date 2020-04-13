#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

#
# Various other C programs/libraries need to be installed on the local machine,
# in directories listed in the "case" statement below.
#
# Usage:
# 1) 3rd_party.sh --check
# 2) 3rd_party.sh --add <install directory>
#
# A complication is that Java's zip functionality turns symlinks into actual
# files, so we need to ensure libraries are only picked up by the name that the
# dynamic loader will look for, and no symlinks are used.  Otherwise the
# installer becomes very bloated.
#

if [ "x$1" = "x--check" ] ; then
    unset INSTALL_DIR
elif [ "x$1" = "x--add" ] ; then
    if [ $# -lt 2 ] ; then
        echo "Install directory required for --add"
        exit 1
    fi
    INSTALL_DIR="$2"
else
    echo "Action required: --add or --check"
    exit 2
fi

case `uname` in

    Darwin)
        BOOST_LOCATION=/usr/local/lib
        BOOST_COMPILER=clang
        BOOST_EXTENSION=mt-x64-1_71.dylib
        BOOST_LIBRARIES='atomic chrono date_time filesystem iostreams log log_setup program_options regex system thread'
        XML_LOCATION=
        GCC_RT_LOCATION=
        STL_LOCATION=
        ZLIB_LOCATION=
        ;;

    Linux)
        if [ -z "$CPP_CROSS_COMPILE" ] ; then
            BOOST_LOCATION=/usr/local/gcc75/lib
            BOOST_COMPILER=gcc
            if [ `uname -m` = aarch64 ] ; then
                BOOST_ARCH=a64
            else
                BOOST_ARCH=x64
            fi
            BOOST_EXTENSION=mt-${BOOST_ARCH}-1_71.so.1.71.0
            BOOST_LIBRARIES='atomic chrono date_time filesystem iostreams log log_setup program_options regex system thread'
            XML_LOCATION=/usr/local/gcc75/lib
            XML_EXTENSION=.so.2
            GCC_RT_LOCATION=/usr/local/gcc75/lib64
            GCC_RT_EXTENSION=.so.1
            STL_LOCATION=/usr/local/gcc75/lib64
            STL_PREFIX=libstdc++
            STL_EXTENSION=.so.6
            ZLIB_LOCATION=
        elif [ "$CPP_CROSS_COMPILE" = macosx ] ; then
            SYSROOT=/usr/local/sysroot-x86_64-apple-macosx10.13
            BOOST_LOCATION=$SYSROOT/usr/local/lib
            BOOST_COMPILER=clang
            BOOST_EXTENSION=mt-x64-1_71.dylib
            BOOST_LIBRARIES='atomic chrono date_time filesystem iostreams log log_setup program_options regex system thread'
            XML_LOCATION=
            GCC_RT_LOCATION=
            STL_LOCATION=
            ZLIB_LOCATION=
        else
            SYSROOT=/usr/local/sysroot-$CPP_CROSS_COMPILE-linux-gnu
            BOOST_LOCATION=$SYSROOT/usr/local/gcc75/lib
            BOOST_COMPILER=gcc
            if [ "$CPP_CROSS_COMPILE" = aarch64 ] ; then
                BOOST_ARCH=a64
            else
                echo "Cannot cross compile to $CPP_CROSS_COMPILE"
                exit 3
            fi
            BOOST_EXTENSION=mt-${BOOST_ARCH}-1_71.so.1.71.0
            BOOST_LIBRARIES='atomic chrono date_time filesystem iostreams log log_setup program_options regex system thread'
            XML_LOCATION=$SYSROOT/usr/local/gcc75/lib
            XML_EXTENSION=.so.2
            GCC_RT_LOCATION=$SYSROOT/usr/local/gcc75/lib64
            GCC_RT_EXTENSION=.so.1
            STL_LOCATION=$SYSROOT/usr/local/gcc75/lib64
            STL_PREFIX=libstdc++
            STL_EXTENSION=.so.6
            ZLIB_LOCATION=
        fi
        ;;

    MINGW*)
        if [ -z "$LOCAL_DRIVE" ] ; then
            LOCAL_DRIVE=C
        fi
        # These directories are correct for the way our Windows 2012r2 build
        # server is currently set up
        BOOST_LOCATION=/$LOCAL_DRIVE/usr/local/lib
        BOOST_COMPILER=vc
        BOOST_EXTENSION=mt-x64-1_71.dll
        BOOST_LIBRARIES='chrono date_time filesystem iostreams log log_setup program_options regex system thread'
        XML_LOCATION=/$LOCAL_DRIVE/usr/local/bin
        XML_EXTENSION=.dll
        GCC_RT_LOCATION=
        # Read VCBASE from environment if defined, otherwise default to VS Professional 2017
        DEFAULTVCBASE=`cd /$LOCAL_DRIVE && cygpath -m -s "Program Files (x86)/Microsoft Visual Studio/2017/Professional"`
        VCBASE=${VCBASE:-$DEFAULTVCBASE}
        VCVER=`ls -1 /$LOCAL_DRIVE/$VCBASE/VC/Redist/MSVC | tail -1`
        STL_LOCATION=/$LOCAL_DRIVE/$VCBASE/VC/Redist/MSVC/$VCVER/x64/Microsoft.VC141.CRT
        STL_PREFIX=
        STL_EXTENSION=140.dll
        ZLIB_LOCATION=/$LOCAL_DRIVE/usr/local/bin
        ZLIB_EXTENSION=1.dll
        ;;

    *)
        echo `uname 2>&1` "- unsupported operating system"
        exit 4
        ;;

esac

if [ -n "$INSTALL_DIR" ] ; then
    mkdir -p "$INSTALL_DIR"
fi

if [ ! -z "$BOOST_LOCATION" ] ; then
    if ls $BOOST_LOCATION/*boost*$BOOST_COMPILER*$BOOST_EXTENSION >/dev/null ; then
        if [ -n "$INSTALL_DIR" ] ; then
            rm -f $INSTALL_DIR/*boost*$BOOST_COMPILER*$BOOST_EXTENSION
            for LIBRARY in $BOOST_LIBRARIES
            do
                cp $BOOST_LOCATION/*boost*$LIBRARY*$BOOST_COMPILER*$BOOST_EXTENSION $INSTALL_DIR
            done
            chmod u+wx $INSTALL_DIR/*boost*$BOOST_COMPILER*$BOOST_EXTENSION
        fi
    else
        echo "Boost libraries not found"
        exit 5
    fi
fi
if [ ! -z "$XML_LOCATION" ] ; then
    if ls $XML_LOCATION/libxml2*$XML_EXTENSION >/dev/null ; then
        if [ -n "$INSTALL_DIR" ] ; then
            rm -f $INSTALL_DIR/libxml2*$XML_EXTENSION
            cp $XML_LOCATION/libxml2*$XML_EXTENSION $INSTALL_DIR
            chmod u+wx $INSTALL_DIR/libxml2*$XML_EXTENSION
        fi
    else
        echo "libxml2 not found"
        exit 6
    fi
fi
if [ ! -z "$GCC_RT_LOCATION" ] ; then
    if ls $GCC_RT_LOCATION/libgcc_s*$GCC_RT_EXTENSION >/dev/null ; then
        if [ -n "$INSTALL_DIR" ] ; then
            rm -f $INSTALL_DIR/libgcc_s*$GCC_RT_EXTENSION
            cp $GCC_RT_LOCATION/libgcc_s*$GCC_RT_EXTENSION $INSTALL_DIR
            chmod u+wx $INSTALL_DIR/libgcc_s*$GCC_RT_EXTENSION
        fi
    else
        echo "gcc runtime library not found"
        exit 7
    fi
fi
if [ ! -z "$STL_LOCATION" ] ; then
    if ls $STL_LOCATION/$STL_PREFIX*$STL_EXTENSION >/dev/null ; then
        if [ -n "$INSTALL_DIR" ] ; then
            rm -f $INSTALL_DIR/$STL_PREFIX*$STL_EXTENSION
            cp $STL_LOCATION/$STL_PREFIX*$STL_EXTENSION $INSTALL_DIR
            chmod u+wx $INSTALL_DIR/$STL_PREFIX*$STL_EXTENSION
        fi
    else
        echo "C++ standard library not found"
        exit 8
    fi
fi
if [ ! -z "$ZLIB_LOCATION" ] ; then
    if ls $ZLIB_LOCATION/zlib*$ZLIB_EXTENSION >/dev/null ; then
        if [ -n "$INSTALL_DIR" ] ; then
            rm -f $INSTALL_DIR/zlib*$ZLIB_EXTENSION
            cp $ZLIB_LOCATION/zlib*$ZLIB_EXTENSION $INSTALL_DIR
            chmod u+wx $INSTALL_DIR/zlib*$ZLIB_EXTENSION
        fi
    else
        echo "zlib not found"
        exit 9
    fi
fi

# Special extra platform-specific processing
case `uname` in

    Linux)
        if [ -n "$INSTALL_DIR" -a "$CPP_CROSS_COMPILE" != macosx ] ; then
            cd "$INSTALL_DIR"
            for FILE in `find . -type f | egrep -v '^core|-debug$|libMl'`
            do
                # Replace RPATH for 3rd party libraries that already have one
                patchelf --print-rpath $FILE | grep lib >/dev/null 2>&1 && patchelf --set-rpath '$ORIGIN/.' $FILE
                if [ $? -eq 0 ] ; then
                    echo "Set RPATH in $FILE"
                else
                    echo "Did not set RPATH in $FILE"
                fi
            done
        fi
        ;;

esac

