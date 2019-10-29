/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CNamedPipeFactory.h>
#include <core/COsFileFuncs.h>
#include <core/CSleep.h>
#include <core/CThread.h>

#include <boost/test/unit_test.hpp>

#include <fstream>

#include <stdio.h>
#ifndef Windows
#include <unistd.h>
#endif

BOOST_AUTO_TEST_SUITE(CNamedPipeFactoryTest)

namespace {

const uint32_t SLEEP_TIME_MS = 100;
const uint32_t PAUSE_TIME_MS = 10;
const size_t MAX_ATTEMPTS = 100;
const size_t TEST_SIZE = 10000;
const char TEST_CHAR = 'a';
#ifdef Windows
const char* const TEST_PIPE_NAME = "\\\\.\\pipe\\testpipe";
#else
const char* const TEST_PIPE_NAME = "testfiles/testpipe";
#endif

class CThreadDataWriter : public ml::core::CThread {
public:
    CThreadDataWriter(const std::string& fileName, size_t size)
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

class CThreadDataReader : public ml::core::CThread {
public:
    CThreadDataReader(const std::string& fileName) : m_FileName(fileName) {}

    const std::string& data() const { return m_Data; }

protected:
    virtual void run() {
        m_Data.clear();

        std::ifstream strm;

        // Try to open the file repeatedly to allow time for the other
        // thread to create it
        size_t attempt(1);
        do {
            BOOST_TEST_REQUIRE(attempt++ <= MAX_ATTEMPTS);
            ml::core::CSleep::sleep(PAUSE_TIME_MS);
            strm.open(m_FileName.c_str());
        } while (!strm.is_open());

        static const std::streamsize BUF_SIZE = 512;
        char buffer[BUF_SIZE];
        while (strm.good()) {
            strm.read(buffer, BUF_SIZE);
            BOOST_TEST_REQUIRE(!strm.bad());
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

class CThreadBlockCanceller : public ml::core::CThread {
public:
    CThreadBlockCanceller(ml::core::CThread::TThreadId threadId)
        : m_ThreadId(threadId) {}

protected:
    virtual void run() {
        // Wait for the file to exist
        ml::core::CSleep::sleep(SLEEP_TIME_MS);

        // Cancel the open() or read() operation on the file
        BOOST_TEST_REQUIRE(ml::core::CThread::cancelBlockedIo(m_ThreadId));
    }

    virtual void shutdown() {}

private:
    ml::core::CThread::TThreadId m_ThreadId;
};
}

BOOST_AUTO_TEST_CASE(testServerIsCppReader) {
    CThreadDataWriter threadWriter(TEST_PIPE_NAME, TEST_SIZE);
    BOOST_TEST_REQUIRE(threadWriter.start());

    ml::core::CNamedPipeFactory::TIStreamP strm =
        ml::core::CNamedPipeFactory::openPipeStreamRead(TEST_PIPE_NAME);
    BOOST_TEST_REQUIRE(strm);

    static const std::streamsize BUF_SIZE = 512;
    std::string readData;
    char buffer[BUF_SIZE];
    do {
        strm->read(buffer, BUF_SIZE);
        BOOST_TEST_REQUIRE(!strm->bad());
        if (strm->gcount() > 0) {
            readData.append(buffer, static_cast<size_t>(strm->gcount()));
        }
    } while (!strm->eof());

    BOOST_REQUIRE_EQUAL(TEST_SIZE, readData.length());
    BOOST_REQUIRE_EQUAL(std::string(TEST_SIZE, TEST_CHAR), readData);

    BOOST_TEST_REQUIRE(threadWriter.stop());

    strm.reset();
}

BOOST_AUTO_TEST_CASE(testServerIsCReader) {
    CThreadDataWriter threadWriter(TEST_PIPE_NAME, TEST_SIZE);
    BOOST_TEST_REQUIRE(threadWriter.start());

    ml::core::CNamedPipeFactory::TFileP file =
        ml::core::CNamedPipeFactory::openPipeFileRead(TEST_PIPE_NAME);
    BOOST_TEST_REQUIRE(file);

    static const size_t BUF_SIZE = 512;
    std::string readData;
    char buffer[BUF_SIZE];
    do {
        size_t charsRead = ::fread(buffer, sizeof(char), BUF_SIZE, file.get());
        BOOST_TEST_REQUIRE(!::ferror(file.get()));
        if (charsRead > 0) {
            readData.append(buffer, charsRead);
        }
    } while (!::feof(file.get()));

    BOOST_REQUIRE_EQUAL(TEST_SIZE, readData.length());
    BOOST_REQUIRE_EQUAL(std::string(TEST_SIZE, TEST_CHAR), readData);

    BOOST_TEST_REQUIRE(threadWriter.stop());

    file.reset();
}

BOOST_AUTO_TEST_CASE(testServerIsCppWriter) {
    CThreadDataReader threadReader(TEST_PIPE_NAME);
    BOOST_TEST_REQUIRE(threadReader.start());

    ml::core::CNamedPipeFactory::TOStreamP strm =
        ml::core::CNamedPipeFactory::openPipeStreamWrite(TEST_PIPE_NAME);
    BOOST_TEST_REQUIRE(strm);

    size_t charsLeft(TEST_SIZE);
    size_t blockSize(7);
    while (charsLeft > 0) {
        if (blockSize > charsLeft) {
            blockSize = charsLeft;
        }
        (*strm) << std::string(blockSize, TEST_CHAR);
        BOOST_TEST_REQUIRE(!strm->bad());
        charsLeft -= blockSize;
    }

    strm.reset();

    BOOST_TEST_REQUIRE(threadReader.stop());

    BOOST_REQUIRE_EQUAL(TEST_SIZE, threadReader.data().length());
    BOOST_REQUIRE_EQUAL(std::string(TEST_SIZE, TEST_CHAR), threadReader.data());
}

BOOST_AUTO_TEST_CASE(testServerIsCWriter) {
    CThreadDataReader threadReader(TEST_PIPE_NAME);
    BOOST_TEST_REQUIRE(threadReader.start());

    ml::core::CNamedPipeFactory::TFileP file =
        ml::core::CNamedPipeFactory::openPipeFileWrite(TEST_PIPE_NAME);
    BOOST_TEST_REQUIRE(file);

    size_t charsLeft(TEST_SIZE);
    size_t blockSize(7);
    while (charsLeft > 0) {
        if (blockSize > charsLeft) {
            blockSize = charsLeft;
        }
        BOOST_TEST_REQUIRE(
            ::fputs(std::string(blockSize, TEST_CHAR).c_str(), file.get()) >= 0);
        charsLeft -= blockSize;
    }

    file.reset();

    BOOST_TEST_REQUIRE(threadReader.stop());

    BOOST_REQUIRE_EQUAL(TEST_SIZE, threadReader.data().length());
    BOOST_REQUIRE_EQUAL(std::string(TEST_SIZE, TEST_CHAR), threadReader.data());
}

BOOST_AUTO_TEST_CASE(testCancelBlock) {
    CThreadBlockCanceller cancellerThread(ml::core::CThread::currentThreadId());
    BOOST_TEST_REQUIRE(cancellerThread.start());

    ml::core::CNamedPipeFactory::TOStreamP strm =
        ml::core::CNamedPipeFactory::openPipeStreamWrite(TEST_PIPE_NAME);
    BOOST_TEST_REQUIRE(strm == nullptr);

    BOOST_TEST_REQUIRE(cancellerThread.stop());
}

BOOST_AUTO_TEST_CASE(testErrorIfRegularFile) {
    ml::core::CNamedPipeFactory::TIStreamP strm =
        ml::core::CNamedPipeFactory::openPipeStreamRead("Main.cc");
    BOOST_TEST_REQUIRE(strm == nullptr);
}

BOOST_AUTO_TEST_CASE(testErrorIfSymlink) {
#ifdef Windows
    // It's impossible to create a symlink to a named pipe on Windows - they
    // live under \\.\pipe\ and it's not possible to symlink to this part of
    // the file system
    LOG_DEBUG(<< "symlink test not relevant to Windows");
    // Suppress the error about no assertions in this case
    BOOST_REQUIRE(BOOST_IS_DEFINED(Windows));
#else
    static const char* const TEST_SYMLINK_NAME = "test_symlink";

    // Remove any files left behind by a previous failed test, but don't check
    // the return codes as these calls will usually fail
    ::unlink(TEST_SYMLINK_NAME);
    ::unlink(TEST_PIPE_NAME);

    BOOST_REQUIRE_EQUAL(0, ::mkfifo(TEST_PIPE_NAME, S_IRUSR | S_IWUSR));
    BOOST_REQUIRE_EQUAL(0, ::symlink(TEST_PIPE_NAME, TEST_SYMLINK_NAME));

    ml::core::CNamedPipeFactory::TIStreamP strm =
        ml::core::CNamedPipeFactory::openPipeStreamRead(TEST_SYMLINK_NAME);
    BOOST_TEST_REQUIRE(strm == nullptr);

    BOOST_REQUIRE_EQUAL(0, ::unlink(TEST_SYMLINK_NAME));
    BOOST_REQUIRE_EQUAL(0, ::unlink(TEST_PIPE_NAME));
#endif
}

BOOST_AUTO_TEST_SUITE_END()
