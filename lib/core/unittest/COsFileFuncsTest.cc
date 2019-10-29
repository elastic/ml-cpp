/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/COsFileFuncs.h>
#include <core/CTimeUtils.h>
#include <core/WindowsSafe.h>

#include <boost/test/unit_test.hpp>

#include <stdio.h>
#include <string.h>
#ifndef Windows
#include <unistd.h>
#endif

BOOST_AUTO_TEST_SUITE(COsFileFuncsTest)

BOOST_AUTO_TEST_CASE(testInode) {
    // Windows doesn't have inodes as such, but on NTFS we can simulate a number
    // that fulfils the purpose of determining when a file has been renamed and
    // another one with the original name has been created.

    ml::core::COsFileFuncs::TStat statBuf;

    std::string mainFile("Main.cc");
    std::string testFile("COsFileFuncsTest.cc");

    ::memset(&statBuf, 0, sizeof(statBuf));
    ml::core::COsFileFuncs::TIno headerDirect(0);
    BOOST_REQUIRE_EQUAL(0, ml::core::COsFileFuncs::stat(mainFile.c_str(), &statBuf));
    headerDirect = statBuf.st_ino;
    LOG_DEBUG(<< "Inode for " << mainFile << " from directory is " << headerDirect);

    ::memset(&statBuf, 0, sizeof(statBuf));
    ml::core::COsFileFuncs::TIno headerOpen(0);
    int headerFd(ml::core::COsFileFuncs::open(mainFile.c_str(), ml::core::COsFileFuncs::RDONLY));
    BOOST_TEST_REQUIRE(headerFd != -1);
    BOOST_REQUIRE_EQUAL(0, ml::core::COsFileFuncs::fstat(headerFd, &statBuf));
    BOOST_REQUIRE_EQUAL(0, ml::core::COsFileFuncs::close(headerFd));
    headerOpen = statBuf.st_ino;
    LOG_DEBUG(<< "Inode for " << mainFile << " from open file is " << headerOpen);

    ::memset(&statBuf, 0, sizeof(statBuf));
    ml::core::COsFileFuncs::TIno implDirect(0);
    BOOST_REQUIRE_EQUAL(0, ml::core::COsFileFuncs::stat(testFile.c_str(), &statBuf));
    implDirect = statBuf.st_ino;
    LOG_DEBUG(<< "Inode for " << testFile << " from directory is " << implDirect);

    ::memset(&statBuf, 0, sizeof(statBuf));
    ml::core::COsFileFuncs::TIno implOpen(0);
    int implFd(ml::core::COsFileFuncs::open(testFile.c_str(), ml::core::COsFileFuncs::RDONLY));
    BOOST_TEST_REQUIRE(implFd != -1);
    BOOST_REQUIRE_EQUAL(0, ml::core::COsFileFuncs::fstat(implFd, &statBuf));
    BOOST_REQUIRE_EQUAL(0, ml::core::COsFileFuncs::close(implFd));
    implOpen = statBuf.st_ino;
    LOG_DEBUG(<< "Inode for " << testFile << " from open file is " << implOpen);

    BOOST_REQUIRE_EQUAL(headerDirect, headerOpen);
    BOOST_REQUIRE_EQUAL(implDirect, implOpen);
    BOOST_TEST_REQUIRE(implDirect != headerDirect);
}

BOOST_AUTO_TEST_CASE(testLStat) {
    std::string file("Main.cc");
    std::string symLink("Main.symlink.cc");

    // Remove any link left behind by a previous failed test, but don't check
    // the return code as this will usually fail
    ::remove(symLink.c_str());

#if defined(NTDDI_VERSION) && NTDDI_VERSION < 0x0A000004
    // Prior to Windows 10 Fall Creator's Update only administrators could
    // create symlinks on Windows, and we don't want to force the unit tests to
    // run as administrator
    LOG_WARN(<< "Skipping lstat() test as it would need to run as administrator");
    // NTDDI_VERSION should only be defined on Windows
    BOOST_REQUIRE(BOOST_IS_DEFINED(Windows));
#else
#ifdef Windows
    BOOST_TEST_REQUIRE(CreateSymbolicLink(symLink.c_str(), file.c_str(),
                                          SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE) != FALSE);
#else
    BOOST_REQUIRE_EQUAL(0, ::symlink(file.c_str(), symLink.c_str()));
#endif

    ml::core::COsFileFuncs::TStat statBuf;
    ::memset(&statBuf, 0, sizeof(statBuf));
    BOOST_REQUIRE_EQUAL(0, ml::core::COsFileFuncs::lstat(symLink.c_str(), &statBuf));
    // Windows doesn't have a flag for symlinks, so just assert that lstat()
    // doesn't think the link is one of the other types of file system object
    BOOST_TEST_REQUIRE((statBuf.st_mode & S_IFMT) != S_IFREG);
    BOOST_TEST_REQUIRE((statBuf.st_mode & S_IFMT) != S_IFDIR);
    BOOST_TEST_REQUIRE((statBuf.st_mode & S_IFMT) != S_IFCHR);

    // Due to the way this test is structured, the link should have been created
    // in the last few seconds (but the linked file, Main.cc, could be older)
    ml::core_t::TTime now = ml::core::CTimeUtils::now();
    LOG_INFO(<< "now: " << now << ", symlink create time: " << statBuf.st_ctime
             << ", symlink modification time: " << statBuf.st_mtime
             << ", symlink access time: " << statBuf.st_atime);
    BOOST_TEST_REQUIRE(statBuf.st_ctime > now - 3);
    BOOST_TEST_REQUIRE(statBuf.st_mtime > now - 3);
    BOOST_TEST_REQUIRE(statBuf.st_atime > now - 3);

    BOOST_REQUIRE_EQUAL(0, ::remove(symLink.c_str()));
#endif
}

BOOST_AUTO_TEST_SUITE_END()
