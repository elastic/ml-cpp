/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include <core/COsFileFuncs.h>

#include <core/WindowsSafe.h>

#include <ctype.h>
#include <direct.h>
#include <errno.h>
#include <fcntl.h>

namespace {

//! Convert from the time structure Windows uses for timestamping files to epoch
//! seconds.  A FILETIME structure stores the number of 100ns ticks since
//! midnight on 1/1/1601 UTC (Gregorian Calendar even though many countries were
//! still using the Julian Calendar then).
__time64_t fileTimeToTimeT(const FILETIME& fileTime) {
    static const ULONGLONG TICKS_PER_SECOND = 10000000ull;
    static const __time64_t SECONDS_1601_TO_1970 = 11644473600ll;
    ULARGE_INTEGER largeInt;
    largeInt.LowPart = fileTime.dwLowDateTime;
    largeInt.HighPart = fileTime.dwHighDateTime;
    return static_cast<__time64_t>(largeInt.QuadPart / TICKS_PER_SECOND) - SECONDS_1601_TO_1970;
}
}

namespace ml {
namespace core {

const int COsFileFuncs::APPEND(O_APPEND);
const int COsFileFuncs::BINARY(O_BINARY);
const int COsFileFuncs::CREAT(O_CREAT);
const int COsFileFuncs::EXCL(O_EXCL);
const int COsFileFuncs::NOFOLLOW(0);
const int COsFileFuncs::RDONLY(O_RDONLY);
const int COsFileFuncs::RDWR(O_RDWR);
const int COsFileFuncs::TEXT(O_TEXT);
const int COsFileFuncs::TRUNC(O_TRUNC);
const int COsFileFuncs::WRONLY(O_WRONLY);
// The number used here mustn't clash with any flags defined in fcntl.h - check
// fcntl.h after upgrading to a newer version of Visual Studio
const int COsFileFuncs::RENAMABLE(0x80000000);

const int COsFileFuncs::EXISTS(0);
const int COsFileFuncs::READABLE(4);
const int COsFileFuncs::WRITABLE(2);
// For Windows, consider "executable" the same as "readable" for the time being
const int COsFileFuncs::EXECUTABLE(4);

const char* COsFileFuncs::NULL_FILENAME("nul");

int COsFileFuncs::open(const char* path, int oflag) {
    return COsFileFuncs::open(path, oflag, 0);
}

int COsFileFuncs::open(const char* path, int oflag, TMode pmode) {
    // To allow the underlying file to be renamed, we have to resort to using
    // the Windows API file functions.  Otherwise we can use the POSIX
    // compatibility layer.
    if ((oflag & RENAMABLE) == 0) {
        // This is the simple case.  Windows won't allow the file to be renamed
        // whilst it's open.
        return ::_open(path, oflag, pmode);
    }

    // Determine the correct access flags
    DWORD desiredAccess(GENERIC_READ);
    if ((oflag & RDWR) != 0) {
        desiredAccess = GENERIC_READ | GENERIC_WRITE;
    } else if ((oflag & WRONLY) != 0) {
        desiredAccess = GENERIC_WRITE;
    }

    DWORD creationDisposition(0);
    switch (oflag & (CREAT | EXCL | TRUNC)) {
    case CREAT:
        creationDisposition = OPEN_ALWAYS;
        break;
    case CREAT | EXCL:
    case CREAT | TRUNC | EXCL:
        creationDisposition = CREATE_NEW;
        break;
    case TRUNC:
        creationDisposition = TRUNCATE_EXISTING;
        break;
    case TRUNC | EXCL:
        // This doesn't make sense
        errno = EINVAL;
        return -1;
    case CREAT | TRUNC:
        creationDisposition = CREATE_ALWAYS;
        break;
    default:
        creationDisposition = OPEN_EXISTING;
        break;
    }

    DWORD attributes(FILE_ATTRIBUTE_NORMAL);
    if ((oflag & CREAT) != 0 && (pmode & S_IWRITE) == 0) {
        attributes = FILE_ATTRIBUTE_READONLY;
    }

    HANDLE handle = CreateFile(path, desiredAccess,
                               FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
                               0, creationDisposition, attributes, 0);
    if (handle == INVALID_HANDLE_VALUE) {
        switch (GetLastError()) {
        case ERROR_FILE_NOT_FOUND:
        case ERROR_PATH_NOT_FOUND:
        case ERROR_INVALID_DRIVE:
        case ERROR_BAD_PATHNAME:
            errno = ENOENT;
            break;
        case ERROR_TOO_MANY_OPEN_FILES:
            errno = EMFILE;
            break;
        case ERROR_ACCESS_DENIED:
        case ERROR_NETWORK_ACCESS_DENIED:
        case ERROR_LOCK_VIOLATION:
        case ERROR_DRIVE_LOCKED:
            errno = EACCES;
            break;
        case ERROR_INVALID_HANDLE:
            errno = EBADF;
            break;
        case ERROR_NOT_ENOUGH_MEMORY:
            errno = ENOMEM;
            break;
        case ERROR_DISK_FULL:
            errno = ENOSPC;
            break;
        default:
            errno = EINVAL;
            break;
        }
        return -1;
    }

    // Convert the Windows handle to a POSIX compatibility layer file descriptor
    int filteredFlags(oflag & (TEXT | RDONLY | APPEND));
    return ::_open_osfhandle(reinterpret_cast<intptr_t>(handle), filteredFlags);
}

int COsFileFuncs::dup(int fildes) {
    return ::_dup(fildes);
}

int COsFileFuncs::dup2(int fildes, int fildes2) {
    return ::_dup2(fildes, fildes2);
}

COsFileFuncs::TOffset COsFileFuncs::lseek(int fildes, TOffset offset, int whence) {
    return ::_lseeki64(fildes, offset, whence);
}

COsFileFuncs::TSignedSize COsFileFuncs::read(int fildes, void* buf, size_t nbyte) {
    return ::_read(fildes, buf, static_cast<unsigned int>(nbyte));
}

COsFileFuncs::TSignedSize COsFileFuncs::write(int fildes, const void* buf, size_t nbyte) {
    return ::_write(fildes, buf, static_cast<unsigned int>(nbyte));
}

int COsFileFuncs::close(int fildes) {
    return ::_close(fildes);
}

int COsFileFuncs::fstat(int fildes, TStat* buf) {
    struct _stati64 tmpBuf;
    int res(::_fstati64(fildes, &tmpBuf));
    if (res != 0) {
        return res;
    }

    // Copy the members from the temporary stat structure to our replacement
    // (which has a bigger st_ino member)
    buf->st_dev = tmpBuf.st_dev;
    buf->st_ino = tmpBuf.st_ino;
    buf->st_mode = tmpBuf.st_mode;
    buf->st_nlink = tmpBuf.st_nlink;
    buf->st_uid = tmpBuf.st_uid;
    buf->st_gid = tmpBuf.st_gid;
    buf->st_rdev = tmpBuf.st_rdev;
    buf->st_size = tmpBuf.st_size;
    buf->st_atime = tmpBuf.st_atime;
    buf->st_mtime = tmpBuf.st_mtime;
    buf->st_ctime = tmpBuf.st_ctime;

    // By default, Windows always sets the st_ino member to 0 - try to do
    // something better
    HANDLE handle(reinterpret_cast<HANDLE>(::_get_osfhandle(fildes)));
    if (handle == INVALID_HANDLE_VALUE) {
        return -1;
    }

    BY_HANDLE_FILE_INFORMATION info;
    if (GetFileInformationByHandle(handle, &info) == FALSE) {
        errno = EACCES;
        return -1;
    }

    buf->st_ino = static_cast<TIno>(info.nFileIndexLow) |
                  (static_cast<TIno>(info.nFileIndexHigh) << 32);

    return 0;
}

int COsFileFuncs::stat(const char* path, TStat* buf) {
    struct _stati64 tmpBuf;
    int res(::_stati64(path, &tmpBuf));
    if (res != 0) {
        return res;
    }

    // Copy the members from the temporary stat structure to our replacement
    // (which has a bigger st_ino member)
    buf->st_dev = tmpBuf.st_dev;
    buf->st_ino = tmpBuf.st_ino;
    buf->st_mode = tmpBuf.st_mode;
    buf->st_nlink = tmpBuf.st_nlink;
    buf->st_uid = tmpBuf.st_uid;
    buf->st_gid = tmpBuf.st_gid;
    buf->st_rdev = tmpBuf.st_rdev;
    buf->st_size = tmpBuf.st_size;
    buf->st_atime = tmpBuf.st_atime;
    buf->st_mtime = tmpBuf.st_mtime;
    buf->st_ctime = tmpBuf.st_ctime;

    // If we're dealing with something other than a normal file, we're done
    if ((buf->st_mode & _S_IFMT) != _S_IFREG) {
        return res;
    }

    // To set st_ino, we have to briefly open the file
    HANDLE handle = CreateFile(path,
                               0, // Open for neither read nor write
                               FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
                               0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    if (handle == INVALID_HANDLE_VALUE) {
        errno = EACCES;
        return -1;
    }

    BY_HANDLE_FILE_INFORMATION info;
    if (GetFileInformationByHandle(handle, &info) == FALSE) {
        CloseHandle(handle);

        errno = EACCES;
        return -1;
    }

    buf->st_ino = static_cast<TIno>(info.nFileIndexLow) |
                  (static_cast<TIno>(info.nFileIndexHigh) << 32);

    CloseHandle(handle);

    return 0;
}

int COsFileFuncs::lstat(const char* path, TStat* buf) {
    // Windows has no lstat() function, but it's only different to stat() in the
    // case where the path points at a symlink, so often we can simply call
    // stat()
    WIN32_FILE_ATTRIBUTE_DATA attributes = {0};
    if (path == nullptr || buf == nullptr ||
        GetFileAttributesEx(path, GetFileExInfoStandard, &attributes) == FALSE ||
        (attributes.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0) {
        return COsFileFuncs::stat(path, buf);
    }

    // This logic for getting the drive number isn't great, but it's exactly
    // what Microsoft's stat() function does!
    int drive = (path[1] == ':') ? (::tolower(path[0]) - 'a' + 1) : ::_getdrive();

    buf->st_dev = drive;
    buf->st_ino = 0;
    // Adopt Unix convention that a symlink has type 012 and 0777 permissions
    // (realistically this just needs to fail to be identified as a regular file
    // or directory)
    buf->st_mode = 0120777u;
    buf->st_nlink = 1;
    // Again, match the behaviour of Microsoft's stat(), i.e. don't attempt to
    // return user/group information
    buf->st_uid = 0;
    buf->st_gid = 0;
    buf->st_rdev = drive;
    buf->st_size = 0;
    buf->st_atime = fileTimeToTimeT(attributes.ftLastAccessTime);
    buf->st_mtime = fileTimeToTimeT(attributes.ftLastWriteTime);
    buf->st_ctime = fileTimeToTimeT(attributes.ftCreationTime);

    return 0;
}

int COsFileFuncs::access(const char* path, int amode) {
    return ::_access(path, amode);
}

char* COsFileFuncs::getcwd(char* buf, size_t size) {
    return ::_getcwd(buf, static_cast<int>(size));
}

int COsFileFuncs::chdir(const char* path) {
    return ::_chdir(path);
}

int COsFileFuncs::mkdir(const char* path) {
    return ::_mkdir(path);
}
}
}
