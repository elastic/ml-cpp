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
        APR_LOCATION=
        BOOST_LOCATION=/usr/local/lib
        BOOST_COMPILER=clang
        BOOST_EXTENSION=mt-1_65_1.dylib
        BOOST_LIBRARIES='date_time filesystem iostreams program_options regex system thread'
        LOG4CXX_LOCATION=/usr/local/lib
        LOG4CXX_EXTENSION=.10.dylib
        XML_LOCATION=
        EXPAT_LOCATION=
        GCC_RT_LOCATION=
        STL_LOCATION=
        ATOMIC_LOCATION=
        ZLIB_LOCATION=
        ;;

    Linux)
        if [ -z "$CPP_CROSS_COMPILE" ] ; then
            ldd --version 2>&1 | grep musl > /dev/null
            if [ $? -ne 0 ] ; then
                APR_LOCATION=/usr/local/gcc62/lib
                APR_EXTENSION=1.so.0
                BOOST_LOCATION=/usr/local/gcc62/lib
                BOOST_COMPILER=gcc
                BOOST_EXTENSION=mt-1_65_1.so.1.65.1
                BOOST_LIBRARIES='date_time filesystem iostreams program_options regex system thread'
                LOG4CXX_LOCATION=/usr/local/gcc62/lib
                LOG4CXX_EXTENSION=.so.10
                XML_LOCATION=/usr/local/gcc62/lib
                XML_EXTENSION=.so.2
                # Ship the version of expat that came with the apr-util library
                EXPAT_LOCATION=/usr/local/gcc62/lib
                EXPAT_EXTENSION=.so.0
                GCC_RT_LOCATION=/usr/local/gcc62/lib64
                GCC_RT_EXTENSION=.so.1
                STL_LOCATION=/usr/local/gcc62/lib64
                STL_PREFIX=libstdc++
                STL_EXTENSION=.so.6
                ATOMIC_LOCATION=
                ZLIB_LOCATION=
            else
                APR_LOCATION=/usr/local/lib
                APR_EXTENSION=1.so.0
                BOOST_LOCATION=/usr/local/lib
                BOOST_COMPILER=gcc
                BOOST_EXTENSION=mt-1_65_1.so.1.65.1
                BOOST_LIBRARIES='date_time filesystem iostreams program_options regex system thread'
                LOG4CXX_LOCATION=/usr/local/lib
                LOG4CXX_EXTENSION=.so.10
                XML_LOCATION=/usr/local/lib
                XML_EXTENSION=.so.2
                # Ship the version of expat that came with the apr-util library
                EXPAT_LOCATION=/usr/local/lib
                EXPAT_EXTENSION=.so.0
                GCC_RT_LOCATION=
                STL_LOCATION=
                ATOMIC_LOCATION=
                ZLIB_LOCATION=
            fi
        else
            if [ "$CPP_CROSS_COMPILE" = macosx ] ; then
                SYSROOT=/usr/local/sysroot-x86_64-apple-macosx10.10
                APR_LOCATION=
                BOOST_LOCATION=$SYSROOT/usr/local/lib
                BOOST_COMPILER=clang
                BOOST_EXTENSION=mt-1_65_1.dylib
                BOOST_LIBRARIES='date_time filesystem iostreams program_options regex system thread'
                LOG4CXX_LOCATION=$SYSROOT/usr/local/lib
                LOG4CXX_EXTENSION=.10.dylib
                XML_LOCATION=
                EXPAT_LOCATION=
                GCC_RT_LOCATION=
                STL_LOCATION=
                ATOMIC_LOCATION=
                ZLIB_LOCATION=
            else
                echo "Cannot cross compile to $CPP_CROSS_COMPILE"
                exit 3
            fi
        fi
        ;;

    MINGW*)
        if [ -z "$LOCAL_DRIVE" ] ; then
            LOCAL_DRIVE=C
        fi
        # These directories are correct for the way our Windows 2008r2 build
        # server is currently set up
        APR_LOCATION=/$LOCAL_DRIVE/usr/local/bin
        APR_EXTENSION=1.dll
        BOOST_LOCATION=/$LOCAL_DRIVE/usr/local/lib
        BOOST_COMPILER=vc
        BOOST_EXTENSION=mt-1_65_1.dll
        BOOST_LIBRARIES='chrono date_time filesystem iostreams program_options regex system thread'
        LOG4CXX_LOCATION=/$LOCAL_DRIVE/usr/local/bin
        LOG4CXX_EXTENSION=.dll
        XML_LOCATION=/$LOCAL_DRIVE/usr/local/lib
        XML_EXTENSION=.dll
        EXPAT_LOCATION=
        GCC_RT_LOCATION=
        # Read VCBASE from environment if defined, otherwise default to VS Express 2013
        DEFAULTVCBASE=`cd /$LOCAL_DRIVE && cygpath -m -s "Program Files (x86)/Microsoft Visual Studio 12.0"`
        VCBASE=${VCBASE:-$DEFAULTVCBASE}
        STL_LOCATION=/$LOCAL_DRIVE/$VCBASE/VC/redist/x64/Microsoft.VC120.CRT
        STL_PREFIX=msvc
        STL_EXTENSION=.dll
        ATOMIC_LOCATION=
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

if [ ! -z "$APR_LOCATION" ] ; then
    if ls $APR_LOCATION/libapr*$APR_EXTENSION >/dev/null ; then
        if [ -n "$INSTALL_DIR" ] ; then
            rm -f $INSTALL_DIR/libapr*$APR_EXTENSION
            cp $APR_LOCATION/libapr*$APR_EXTENSION $INSTALL_DIR
            chmod u+wx $INSTALL_DIR/libapr*$APR_EXTENSION $INSTALL_DIR
        fi
    else
        echo "Apache Portable Runtime library not found"
        exit 5
    fi
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
        exit 6
    fi
fi
if [ ! -z "$LOG4CXX_LOCATION" ] ; then
    if ls $LOG4CXX_LOCATION/*log4cxx*$LOG4CXX_EXTENSION >/dev/null ; then
        if [ -n "$INSTALL_DIR" ] ; then
            rm -f $INSTALL_DIR/*log4cxx*$LOG4CXX_EXTENSION
            cp $LOG4CXX_LOCATION/*log4cxx*$LOG4CXX_EXTENSION $INSTALL_DIR
            chmod u+wx $INSTALL_DIR/*log4cxx*$LOG4CXX_EXTENSION
        fi
    else
        echo "log4cxx library not found"
        exit 7
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
        exit 8
    fi
fi
if [ ! -z "$EXPAT_LOCATION" ] ; then
    if ls $EXPAT_LOCATION/libexpat*$EXPAT_EXTENSION >/dev/null ; then
        if [ -n "$INSTALL_DIR" ] ; then
            rm -f $INSTALL_DIR/libexpat*$EXPAT_EXTENSION
            cp $EXPAT_LOCATION/libexpat*$EXPAT_EXTENSION $INSTALL_DIR
            chmod u+wx $INSTALL_DIR/libexpat*$EXPAT_EXTENSION
        fi
    else
        echo "Expat library not found"
        exit 9
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
        exit 10
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
        exit 11
    fi
fi
if [ ! -z "$ATOMIC_LOCATION" ] ; then
    if ls $ATOMIC_LOCATION/libstatomic*$ATOMIC_EXTENSION >/dev/null ; then
        if [ -n "$INSTALL_DIR" ] ; then
            rm -f $INSTALL_DIR/libstatomic*$ATOMIC_EXTENSION
            cp $ATOMIC_LOCATION/libstatomic*$ATOMIC_EXTENSION $INSTALL_DIR
            chmod u+wx $INSTALL_DIR/libstatomic*$ATOMIC_EXTENSION
        fi
    else
        echo "Atomic library not found"
        exit 12
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
        exit 13
    fi
fi

# Special extra platform-specific processing
case `uname` in

    Linux)
        if [ -n "$INSTALL_DIR" -a -z "$CPP_CROSS_COMPILE" ] ; then
            cd "$INSTALL_DIR"
            for FILE in `find . -type f | egrep -v '^core|-debug$|libMl'`
            do
                # Replace RPATH for 3rd party libraries that already have one
                patchelf --print-rpath $FILE | grep lib >/dev/null 2>&1 && patchelf --set-rpath '$ORIGIN/.' $FILE
                if [ $? -eq 0 ] ; then
                    echo "Set RPATH in $FILE"
                else
                    # Set RPATH for 3rd party libraries that reference other libraries we ship
                    ldd $FILE | grep /usr/local/lib >/dev/null 2>&1 && patchelf --set-rpath '$ORIGIN/.' $FILE
                    if [ $? -eq 0 ] ; then
                        echo "Set RPATH in $FILE"
                    else
                        echo "Did not set RPATH in $FILE"
                    fi
                fi
            done
        fi
        ;;

    SunOS)
        if [ -n "$INSTALL_DIR" ] ; then
            cd "$INSTALL_DIR"
            for FILE in `find . -type f | egrep -v '^core|-debug$|libMl'`
            do
                # Replace RPATH for 3rd party libraries that already have one
                elfedit -r -e 'dyn:runpath' $FILE >/dev/null 2>&1 && chmod u+wx $FILE && elfedit -e 'dyn:runpath $ORIGIN/.' $FILE
                if [ $? -eq 0 ] ; then
                    echo "Set RPATH in $FILE"
                else
                    # Set RPATH for 3rd party libraries that reference other libraries we ship
                    ldd $FILE | grep /usr/local/lib >/dev/null 2>&1 && chmod u+wx $FILE && elfedit -e 'dyn:runpath $ORIGIN/.' $FILE
                    if [ $? -eq 0 ] ; then
                        echo "Set RPATH in $FILE"
                    else
                        echo "Did not set RPATH in $FILE"
                    fi
                fi
            done
        fi
        ;;

esac

