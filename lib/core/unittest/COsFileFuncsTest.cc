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
#include "COsFileFuncsTest.h"

#include <core/CLogger.h>
#include <core/COsFileFuncs.h>
#include <core/CTimeUtils.h>
#include <core/WindowsSafe.h>

#include <stdio.h>
#include <string.h>
#ifndef Windows
#include <unistd.h>
#endif


CppUnit::Test *COsFileFuncsTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("COsFileFuncsTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<COsFileFuncsTest>(
                                   "COsFileFuncsTest::testInode",
                                   &COsFileFuncsTest::testInode) );
    suiteOfTests->addTest( new CppUnit::TestCaller<COsFileFuncsTest>(
                                   "COsFileFuncsTest::testLStat",
                                   &COsFileFuncsTest::testLStat) );

    return suiteOfTests;
}

void COsFileFuncsTest::testInode(void)
{
    // Windows doesn't have inodes as such, but on NTFS we can simulate a number
    // that fulfils the purpose of determining when a file has been renamed and
    // another one with the original name has been created.

    ml::core::COsFileFuncs::TStat statBuf;

    std::string headerFile("COsFileFuncsTest.h");
    std::string implFile("COsFileFuncsTest.cc");

    ::memset(&statBuf, 0, sizeof(statBuf));
    ml::core::COsFileFuncs::TIno headerDirect(0);
    CPPUNIT_ASSERT_EQUAL(0, ml::core::COsFileFuncs::stat(headerFile.c_str(),
                                                         &statBuf));
    headerDirect = statBuf.st_ino;
    LOG_DEBUG("Inode for " << headerFile << " from directory is " <<
              headerDirect);

    ::memset(&statBuf, 0, sizeof(statBuf));
    ml::core::COsFileFuncs::TIno headerOpen(0);
    int headerFd(ml::core::COsFileFuncs::open(headerFile.c_str(),
                                              ml::core::COsFileFuncs::RDONLY));
    CPPUNIT_ASSERT(headerFd != -1);
    CPPUNIT_ASSERT_EQUAL(0, ml::core::COsFileFuncs::fstat(headerFd,
                                                          &statBuf));
    CPPUNIT_ASSERT_EQUAL(0, ml::core::COsFileFuncs::close(headerFd));
    headerOpen = statBuf.st_ino;
    LOG_DEBUG("Inode for " << headerFile << " from open file is " <<
              headerOpen);

    ::memset(&statBuf, 0, sizeof(statBuf));
    ml::core::COsFileFuncs::TIno implDirect(0);
    CPPUNIT_ASSERT_EQUAL(0, ml::core::COsFileFuncs::stat(implFile.c_str(),
                                                         &statBuf));
    implDirect = statBuf.st_ino;
    LOG_DEBUG("Inode for " << implFile << " from directory is " <<
              implDirect);

    ::memset(&statBuf, 0, sizeof(statBuf));
    ml::core::COsFileFuncs::TIno implOpen(0);
    int implFd(ml::core::COsFileFuncs::open(implFile.c_str(),
                                            ml::core::COsFileFuncs::RDONLY));
    CPPUNIT_ASSERT(implFd != -1);
    CPPUNIT_ASSERT_EQUAL(0, ml::core::COsFileFuncs::fstat(implFd,
                                                          &statBuf));
    CPPUNIT_ASSERT_EQUAL(0, ml::core::COsFileFuncs::close(implFd));
    implOpen = statBuf.st_ino;
    LOG_DEBUG("Inode for " << implFile << " from open file is " <<
              implOpen);

    CPPUNIT_ASSERT_EQUAL(headerDirect, headerOpen);
    CPPUNIT_ASSERT_EQUAL(implDirect, implOpen);
    CPPUNIT_ASSERT(implDirect != headerDirect);
}

void COsFileFuncsTest::testLStat(void)
{
    std::string file("Main.cc");
    std::string symLink("Main.symlink.cc");

    // Remove any link left behind by a previous failed test, but don't check
    // the return code as this will usually fail
    ::remove(symLink.c_str());

#if defined(NTDDI_VERSION) && NTDDI_VERSION < 0x0A000004
    // Prior to Windows 10 Fall Creator's Update only administrators could
    // create symlinks on Windows, and we don't want to force the unit tests to
    // run as administrator
    LOG_WARN("Skipping lstat() test as it would need to run as administrator");
#else
#ifdef Windows
    CPPUNIT_ASSERT(CreateSymbolicLink(symLink.c_str(),
                                      file.c_str(),
                                      SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE) != FALSE);
#else
    CPPUNIT_ASSERT_EQUAL(0, ::symlink(file.c_str(), symLink.c_str()));
#endif

    ml::core::COsFileFuncs::TStat statBuf;
    ::memset(&statBuf, 0, sizeof(statBuf));
    CPPUNIT_ASSERT_EQUAL(0, ml::core::COsFileFuncs::lstat(symLink.c_str(),
                                                          &statBuf));
    // Windows doesn't have a flag for symlinks, so just assert that lstat()
    // doesn't think the link is one of the other types of file system object
    CPPUNIT_ASSERT((statBuf.st_mode & S_IFMT) != S_IFREG);
    CPPUNIT_ASSERT((statBuf.st_mode & S_IFMT) != S_IFDIR);
    CPPUNIT_ASSERT((statBuf.st_mode & S_IFMT) != S_IFCHR);

    // Due to the way this test is structured, the link should have been created
    // in the last few seconds (but the linked file, Main.cc, could be older)
    ml::core_t::TTime now = ml::core::CTimeUtils::now();
    LOG_INFO("now: " << now <<
             ", symlink create time: " << statBuf.st_ctime <<
             ", symlink modification time: " << statBuf.st_mtime <<
             ", symlink access time: " << statBuf.st_atime);
    CPPUNIT_ASSERT(statBuf.st_ctime > now - 3);
    CPPUNIT_ASSERT(statBuf.st_mtime > now - 3);
    CPPUNIT_ASSERT(statBuf.st_atime > now - 3);

    CPPUNIT_ASSERT_EQUAL(0, ::remove(symLink.c_str()));
#endif
}

