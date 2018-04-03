/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_COsFileFuncs_h
#define INCLUDED_ml_core_COsFileFuncs_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#ifdef Windows
#include <io.h>
#include <stdint.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#ifndef Windows
#include <unistd.h>
#endif


namespace ml
{
namespace core
{


//! \brief
//! Portable wrapper around OS level file functions.
//!
//! DESCRIPTION:\n
//! Portable wrapper around OS level file functions, as defined
//! in unistd.h on Unix or io.h on Windows.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The methods of this class have names and signatures identical to
//! the Unix versions of the functions they're making portable.  They
//! also have behaviour as close as possible to these Unix functions.
//! This means:
//! a) The names don't quite always adhere to the coding standards
//! b) They're very much old style C functions in their interface
//! Further wrapping should be applied to make this pure portability
//! layer more C++-like and more Ml-like.
//!
//! On Unix sizes are left to the system headers, i.e. we'd define
//! _LARGEFILE_SOURCE and let the macros in the system headers decide
//! whether to use 64 bit or 32 bit types for file sizes, offsets, etc.
//!
//! On Windows, we have to explicitly request 64 bit types in the
//! typedefs and functions in this class.
//!
class CORE_EXPORT COsFileFuncs : private CNonInstantiatable
{
    public:
        //! Use in place of OS level file flags - will be defined as zero on
        //! platforms that don't support them
        static const int APPEND;
        static const int BINARY;
        static const int CREAT;
        static const int EXCL;
        static const int NOFOLLOW;
        static const int RDONLY;
        static const int RDWR;
        static const int TEXT;
        static const int TRUNC;
        static const int WRONLY;
        static const int RENAMABLE;

        //! Use in place of OS level access flags - will be defined as zero on
        //! platforms that don't support them
        static const int EXISTS;
        static const int READABLE;
        static const int WRITABLE;
        static const int EXECUTABLE;

        //! The name of the magic file that discards everything written to it
        static const char *NULL_FILENAME;

    public:
        //! Signed size type (to be used instead of ssize_t)
#ifdef Windows
        using TSignedSize = int;
#else
        using TSignedSize = ssize_t;
#endif

        //! Offset type (to be used instead of off_t)
#ifdef Windows
        using TOffset = __int64;
#else
        using TOffset = off_t;
#endif

        //! Mode type (to be used instead of mode_t)
#ifdef Windows
        using TMode = int;
#else
        using TMode = mode_t;
#endif

        //! Inode type (to be used instead of ino_t)
#ifdef Windows
        using TIno = uint64_t;
#else
        using TIno = ino_t;
#endif

        //! Stat buffer struct (to be used instead of struct stat)
#ifdef Windows
        struct SStat
        {
            // Member names don't conform to the coding standards because they
            // need to match those of struct stat
            _dev_t         st_dev;
            //! Replaces the _ino_t member of _stati64
            TIno           st_ino;
            unsigned short st_mode;
            short          st_nlink;
            short          st_uid;
            short          st_gid;
            _dev_t         st_rdev;
            __int64        st_size;
            __time64_t     st_atime;
            __time64_t     st_mtime;
            __time64_t     st_ctime;
        };

        using TStat = SStat;
#else
        using TStat = struct stat;
#endif

    public:
        static int open(const char *path, int oflag);
        static int open(const char *path, int oflag, TMode pmode);
        static int dup(int fildes);
        static int dup2(int fildes, int fildes2);
        static TOffset lseek(int fildes, TOffset offset, int whence);
        static TSignedSize read(int fildes, void *buf, size_t nbyte);
        static TSignedSize write(int fildes, const void *buf, size_t nbyte);
        static int close(int fildes);

        static int fstat(int fildes, TStat *buf);
        static int stat(const char *path, TStat *buf);
        static int lstat(const char *path, TStat *buf);
        static int access(const char *path, int amode);

        static char *getcwd(char *buf, size_t size);
        static int chdir(const char *path);
        static int mkdir(const char *path);
};


}
}

#endif // INCLUDED_ml_core_COsFileFuncs_h

