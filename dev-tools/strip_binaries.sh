#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Strips ML native code binaries to make them smaller before distribution.

ML_APP_NAME=controller
case `uname` in

    Darwin)
        EXE_DIR="$ML_APP_NAME.app/Contents/MacOS"
        DYNAMIC_LIB_DIR="$ML_APP_NAME.app/Contents/lib"
        ;;

    Linux)
        if [ "$CPP_CROSS_COMPILE" = macosx ] ; then
            EXE_DIR="$ML_APP_NAME.app/Contents/MacOS"
            DYNAMIC_LIB_DIR="$ML_APP_NAME.app/Contents/lib"
        else
            EXE_DIR=bin
            DYNAMIC_LIB_DIR=lib
        fi
        ;;

esac

# Ensure $CPP_PLATFORM_HOME is set
if [ -z "$CPP_PLATFORM_HOME" ] ; then
    echo '$CPP_PLATFORM_HOME is not set'
    exit 1
fi

# Ensure the executable programs folder has been created.
if [ ! -d "$CPP_PLATFORM_HOME/$EXE_DIR" ] ; then
    echo "$CPP_PLATFORM_HOME/$EXE_DIR does not exist"
    exit 2
fi

# Ensure the lib folder has been created.
if [ ! -d "$CPP_PLATFORM_HOME/$DYNAMIC_LIB_DIR" ] ; then
    echo "$CPP_PLATFORM_HOME/$DYNAMIC_LIB_DIR does not exist"
    exit 3
fi

cd "$CPP_PLATFORM_HOME"

# Strip *nix binaries to reduce the download size.  (Stripping is not required
# on Windows, as the symbols are in the .pdb files which we don't ship.)
case `uname` in

    Darwin)
        for PROGRAM in `ls -1d "$EXE_DIR"/* | grep -v '\.dSYM$'`
        do
            echo "Stripping $PROGRAM"
            dsymutil $PROGRAM
            strip -u -r $PROGRAM
        done
        for LIBRARY in `ls -1d "$DYNAMIC_LIB_DIR"/* | grep -v '\.dSYM$'`
        do
            echo "Stripping $LIBRARY"
            case $LIBRARY in
                *Ml*)
                    dsymutil $LIBRARY
            esac
            strip -x $LIBRARY
        done
        ;;

    Linux)
        if [ -z "$CPP_CROSS_COMPILE" ] ; then
            for PROGRAM in `ls -1 "$EXE_DIR"/* | egrep -v "$EXE_DIR"'/core|-debug$'`
            do
                echo "Stripping $PROGRAM"
                objcopy --only-keep-debug "$PROGRAM" "$PROGRAM-debug"
                strip --strip-all $PROGRAM
                objcopy --add-gnu-debuglink="$PROGRAM-debug" "$PROGRAM"
                chmod -x "$PROGRAM-debug"
            done
            for LIBRARY in `ls -1 "$DYNAMIC_LIB_DIR"/* | egrep -v 'lib/core|-debug$'`
            do
                echo "Stripping $LIBRARY"
                objcopy --only-keep-debug "$LIBRARY" "$LIBRARY-debug"
                strip --strip-unneeded $LIBRARY
                objcopy --add-gnu-debuglink="$LIBRARY-debug" "$LIBRARY"
            done
        elif [ "$CPP_CROSS_COMPILE" = macosx ] ; then
            CROSS_TARGET_PLATFORM=x86_64-apple-macosx10.13
            for PROGRAM in `ls -1d "$EXE_DIR"/* | grep -v '\.dSYM$'`
            do
                echo "Stripping $PROGRAM"
                llvm-dsymutil-6.0 $PROGRAM
                /usr/local/bin/$CROSS_TARGET_PLATFORM-strip -u -r $PROGRAM
            done
            for LIBRARY in `ls -1d "$DYNAMIC_LIB_DIR"/* | grep -v '\.dSYM$'`
            do
                echo "Stripping $LIBRARY"
                case $LIBRARY in
                    *Ml*)
                        llvm-dsymutil-6.0 $LIBRARY
                esac
                /usr/local/bin/$CROSS_TARGET_PLATFORM-strip -x $LIBRARY
            done
        else
            CROSS_TARGET_PLATFORM=$CPP_CROSS_COMPILE-linux-gnu
            for PROGRAM in `ls -1 "$EXE_DIR"/* | egrep -v "$EXE_DIR"'/core|-debug$'`
            do
                echo "Stripping $PROGRAM"
                $CROSS_TARGET_PLATFORM-objcopy --only-keep-debug "$PROGRAM" "$PROGRAM-debug"
                $CROSS_TARGET_PLATFORM-strip --strip-all $PROGRAM
                $CROSS_TARGET_PLATFORM-objcopy --add-gnu-debuglink="$PROGRAM-debug" "$PROGRAM"
                chmod -x "$PROGRAM-debug"
            done
            for LIBRARY in `ls -1 "$DYNAMIC_LIB_DIR"/* | egrep -v 'lib/core|-debug$'`
            do
                echo "Stripping $LIBRARY"
                $CROSS_TARGET_PLATFORM-objcopy --only-keep-debug "$LIBRARY" "$LIBRARY-debug"
                $CROSS_TARGET_PLATFORM-strip --strip-unneeded $LIBRARY
                $CROSS_TARGET_PLATFORM-objcopy --add-gnu-debuglink="$LIBRARY-debug" "$LIBRARY"
            done
        fi
        ;;

esac

