/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/COsFileFuncs.h>

#include <fcntl.h>

namespace ml {
namespace core {

const int COsFileFuncs::APPEND(O_APPEND);
const int COsFileFuncs::BINARY(0);
const int COsFileFuncs::CREAT(O_CREAT);
const int COsFileFuncs::EXCL(O_EXCL);
const int COsFileFuncs::NOFOLLOW(O_NOFOLLOW);
const int COsFileFuncs::RDONLY(O_RDONLY);
const int COsFileFuncs::RDWR(O_RDWR);
const int COsFileFuncs::TEXT(0);
const int COsFileFuncs::TRUNC(O_TRUNC);
const int COsFileFuncs::WRONLY(O_WRONLY);
const int COsFileFuncs::RENAMABLE(0);

const int COsFileFuncs::EXISTS(F_OK);
const int COsFileFuncs::READABLE(R_OK);
const int COsFileFuncs::WRITABLE(W_OK);
const int COsFileFuncs::EXECUTABLE(X_OK);

const char* const COsFileFuncs::NULL_FILENAME("/dev/null");

int COsFileFuncs::open(const char* path, int oflag) {
    return ::open(path, oflag);
}

int COsFileFuncs::open(const char* path, int oflag, TMode pmode) {
    return ::open(path, oflag, pmode);
}

int COsFileFuncs::dup(int fildes) {
    return ::dup(fildes);
}

int COsFileFuncs::dup2(int fildes, int fildes2) {
    return ::dup2(fildes, fildes2);
}

COsFileFuncs::TOffset COsFileFuncs::lseek(int fildes, TOffset offset, int whence) {
    return ::lseek(fildes, offset, whence);
}

COsFileFuncs::TSignedSize COsFileFuncs::read(int fildes, void* buf, size_t nbyte) {
    return ::read(fildes, buf, nbyte);
}

COsFileFuncs::TSignedSize COsFileFuncs::write(int fildes, const void* buf, size_t nbyte) {
    return ::write(fildes, buf, nbyte);
}

int COsFileFuncs::close(int fildes) {
    return ::close(fildes);
}

int COsFileFuncs::fstat(int fildes, TStat* buf) {
    return ::fstat(fildes, buf);
}

int COsFileFuncs::stat(const char* path, TStat* buf) {
    return ::stat(path, buf);
}

int COsFileFuncs::lstat(const char* path, TStat* buf) {
    return ::lstat(path, buf);
}

int COsFileFuncs::access(const char* path, int amode) {
    return ::access(path, amode);
}

char* COsFileFuncs::getcwd(char* buf, size_t size) {
    return ::getcwd(buf, size);
}

int COsFileFuncs::chdir(const char* path) {
    return ::chdir(path);
}

int COsFileFuncs::mkdir(const char* path) {
    // Windows doesn't support a mode, hence this method doesn't either.
    // Instead, always create the directory with permissions for the current
    // user only.
    return ::mkdir(path, S_IRUSR | S_IWUSR | S_IXUSR);
}
}
}
