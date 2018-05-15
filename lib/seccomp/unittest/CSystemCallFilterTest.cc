/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CSystemCallFilterTest.h"

#include <core/CLogger.h>
#include <core/COsFileFuncs.h>
#include <core/CSleep.h>
#include <core/CThread.h>

#include <seccomp/CSystemCallFilter.h>

#include <test/CTestTmpDir.h>

#include <cstdlib>
#include <pwd.h>
#include <string>
#include <unistd.h>

namespace {

const uint32_t SLEEP_TIME_MS = 100;
const size_t TEST_SIZE = 100;
const size_t MAX_ATTEMPTS = 20;
const char TEST_CHAR = 'a';
const char* TEST_READ_PIPE_NAME = "testreadpipe";
const char* TEST_WRITE_PIPE_NAME = "testwritepipe";

class CNamedPipeWriter : public ml::core::CThread {
public:
    CNamedPipeWriter(const std::string& fileName, size_t size)
        : m_FileName(fileName), m_Size(size) {}

protected:
    virtual void run() {
        // Wait for the file to exist
        ml::core::CSleep::sleep(SLEEP_TIME_MS);

        std::ofstream strm(m_FileName.c_str());
        for (size_t i = 0; i < m_Size && strm.good(); ++i) {
            strm << TEST_CHAR;
        }
    }

    virtual void shutdown() {}

private:
    std::string m_FileName;
    size_t m_Size;
};

class CNamedPipeReader : public ml::core::CThread {
public:
    CNamedPipeReader(const std::string& fileName) : m_FileName(fileName) {}

    const std::string& data() const { return m_Data; }

protected:
    virtual void run() {
        m_Data.clear();

        std::ifstream strm;

        // Try to open the file repeatedly to allow time for the other
        // thread to create it
        size_t attempt(1);
        do {
            CPPUNIT_ASSERT(attempt++ <= MAX_ATTEMPTS);
            ml::core::CSleep::sleep(SLEEP_TIME_MS);
            strm.open(m_FileName.c_str());
        } while (!strm.is_open());

        static const std::streamsize BUF_SIZE = 512;
        char buffer[BUF_SIZE];
        while (strm.good()) {
            strm.read(buffer, BUF_SIZE);
            CPPUNIT_ASSERT(!strm.bad());
            if (strm.gcount() > 0) {
                // This code deals with the test character we write to
                // detect the short-lived connection problem on Windows
                const char* copyFrom = buffer;
                size_t copyLen = static_cast<size_t>(strm.gcount());
                if (m_Data.empty() && *buffer == ml::core::CNamedPipeFactory::TEST_CHAR) {
                    ++copyFrom;
                    --copyLen;
                }
                if (copyLen > 0) {
                    m_Data.append(copyFrom, copyLen);
                }
            }
        }
    }

    virtual void shutdown() {}

private:
    std::string m_FileName;
    std::string m_Data;
};
}

bool systemCall() {
    return std::system("hostname") == 0;
}

CppUnit::Test* CSystemCallFilterTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CSystemCallFilterTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CSystemCallFilterTest>(
        "CSystemCallFilterTest::testSystemCallFilter",
        &CSystemCallFilterTest::testSystemCallFilter));

    return suiteOfTests;
}

void CSystemCallFilterTest::testSystemCallFilter() {
    // Ensure actions are not prohibited before the
    // system call filters are applied
    CPPUNIT_ASSERT(systemCall());

    // // Install the filter
    ml::seccomp::CSystemCallFilter filter;

    CPPUNIT_ASSERT_ASSERTION_FAIL_MESSAGE("Calling std::system should fail",
                                          CPPUNIT_ASSERT(systemCall()));

    // Operations that must function after seccomp is initialised
    openPipeAndRead(std::string("/private/tmp/") + TEST_READ_PIPE_NAME);
    openPipeAndWrite(std::string("/private/tmp/") + TEST_WRITE_PIPE_NAME);

    // // Write is only possible in /private/tmp, check a pipe cannot
    // // be opened in another dir.
    // const char* homedir;
    // if ((homedir = getenv("HOME")) == nullptr) {
    //     homedir = getpwuid(getuid())->pw_dir;
    // }
    // CPPUNIT_ASSERT_ASSERTION_FAIL_MESSAGE(
    //     "Named pipes cannot be created outside of the designated directory",
    //     assertOpenPipeRead(std::string(homedir) + "/systemcallfilter_unittest_artifact"));

    // CPPUNIT_ASSERT_ASSERTION_FAIL_MESSAGE(
    //     "Named pipes cannot be created outside of the designated directory",
    //     assertOpenPipeWrite(std::string(homedir) + "/systemcallfilter_unittest_artifact"));
}

void CSystemCallFilterTest::openPipeAndRead(const std::string& filename) {

    CNamedPipeWriter threadWriter(filename, TEST_SIZE);
    CPPUNIT_ASSERT(threadWriter.start());

    ml::core::CNamedPipeFactory::TIStreamP strm = assertOpenPipeRead(filename);

    static const std::streamsize BUF_SIZE = 512;
    std::string readData;
    readData.reserve(TEST_SIZE);
    char buffer[BUF_SIZE];
    do {
        strm->read(buffer, BUF_SIZE);
        CPPUNIT_ASSERT(!strm->bad());
        if (strm->gcount() > 0) {
            readData.append(buffer, static_cast<size_t>(strm->gcount()));
        }
    } while (!strm->eof());

    CPPUNIT_ASSERT_EQUAL(TEST_SIZE, readData.length());
    CPPUNIT_ASSERT_EQUAL(std::string(TEST_SIZE, TEST_CHAR), readData);

    CPPUNIT_ASSERT(threadWriter.stop());

    strm.reset();
}

ml::core::CNamedPipeFactory::TIStreamP
CSystemCallFilterTest::assertOpenPipeRead(const std::string& filename) {
    ml::core::CNamedPipeFactory::TIStreamP strm =
        ml::core::CNamedPipeFactory::openPipeStreamRead(filename);
    CPPUNIT_ASSERT(strm);
    return strm;
}

void CSystemCallFilterTest::openPipeAndWrite(const std::string& filename) {
    CNamedPipeReader threadReader(filename);
    CPPUNIT_ASSERT(threadReader.start());

    ml::core::CNamedPipeFactory::TOStreamP strm = assertOpenPipeWrite(filename);

    size_t charsLeft(TEST_SIZE);
    size_t blockSize(7);
    while (charsLeft > 0) {
        if (blockSize > charsLeft) {
            blockSize = charsLeft;
        }
        (*strm) << std::string(blockSize, TEST_CHAR);
        CPPUNIT_ASSERT(!strm->bad());
        charsLeft -= blockSize;
    }

    strm.reset();

    CPPUNIT_ASSERT(threadReader.stop());

    CPPUNIT_ASSERT_EQUAL(TEST_SIZE, threadReader.data().length());
    CPPUNIT_ASSERT_EQUAL(std::string(TEST_SIZE, TEST_CHAR), threadReader.data());
}

ml::core::CNamedPipeFactory::TOStreamP
CSystemCallFilterTest::assertOpenPipeWrite(const std::string& filename) {
    ml::core::CNamedPipeFactory::TOStreamP strm =
        ml::core::CNamedPipeFactory::openPipeStreamWrite(filename);
    CPPUNIT_ASSERT(strm);
    return strm;
}
