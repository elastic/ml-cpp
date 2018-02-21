#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Strips Ml native code binaries to make them smaller before distribution.

# Ensure $CPP_PLATFORM_HOME is set
if [ -z "$CPP_PLATFORM_HOME" ] ; then
    echo '$CPP_PLATFORM_HOME is not set'
    exit 1
fi

# Ensure the bin folder has been created.
if [ ! -d "$CPP_PLATFORM_HOME/bin" ] ; then
    echo '$CPP_PLATFORM_HOME/bin does not exist'
    exit 2
fi

# Ensure the lib folder has been created.
if [ ! -d "$CPP_PLATFORM_HOME/lib" ] ; then
    echo '$CPP_PLATFORM_HOME/lib does not exist'
    exit 3
fi

cd "$CPP_PLATFORM_HOME"

# Strip *nix binaries to reduce the download size.  (Stripping is not required
# on Windows, as the symbols are in the .pdb files which we don't ship.)
case `uname` in

    Darwin)
        for PROGRAM in `ls -1d bin/* | grep -v '\.dSYM$'`
        do
            echo "Stripping $PROGRAM"
            dsymutil $PROGRAM
            strip -u -r $PROGRAM
        done
        for LIBRARY in `ls -1d lib/* | grep -v '\.dSYM$'`
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
            for PROGRAM in `ls -1 bin/* | egrep -v 'bin/core|-debug$'`
            do
                echo "Stripping $PROGRAM"
                objcopy --only-keep-debug "$PROGRAM" "$PROGRAM-debug"
                strip --strip-all $PROGRAM
                objcopy --add-gnu-debuglink="$PROGRAM-debug" "$PROGRAM"
                chmod -x "$PROGRAM-debug"
            done
            for LIBRARY in `ls -1 lib/* | egrep -v 'lib/core|-debug$'`
            do
                echo "Stripping $LIBRARY"
                objcopy --only-keep-debug "$LIBRARY" "$LIBRARY-debug"
                strip --strip-unneeded $LIBRARY
                objcopy --add-gnu-debuglink="$LIBRARY-debug" "$LIBRARY"
            done
        else
            if [ "$CPP_CROSS_COMPILE" = macosx ] ; then
                for PROGRAM in `ls -1d bin/* | grep -v '\.dSYM$'`
                do
                    echo "Stripping $PROGRAM"
                    llvm-dsymutil-3.8 $PROGRAM
                    /usr/local/bin/x86_64-apple-macosx10.10-strip -u -r $PROGRAM
                done
                for LIBRARY in `ls -1d lib/* | grep -v '\.dSYM$'`
                do
                    echo "Stripping $LIBRARY"
                    case $LIBRARY in
                        *Ml*)
                            llvm-dsymutil-3.8 $LIBRARY
                    esac
                    /usr/local/bin/x86_64-apple-macosx10.10-strip -x $LIBRARY
                done
            else
                echo "Cannot cross compile to $CPP_CROSS_COMPILE"
                exit 4
            fi
        fi
        ;;

esac

