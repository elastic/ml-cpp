/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CNamedPipeFactoryTest.h"

#include <core/AtomicTypes.h>
#include <core/CLogger.h>
#include <core/CNamedPipeFactory.h>
#include <core/COsFileFuncs.h>
#include <core/CSleep.h>
#include <core/CThread.h>

#include <fstream>

#include <cstdio>

#ifndef Windows
#include <unistd.h>
#endif

namespace {

const std::uint32_t SLEEP_TIME_MS{100};
const std::uint32_t PAUSE_TIME_MS{10};
const std::size_t MAX_ATTEMPTS{100};
const std::size_t TEST_SIZE{10000};
const char TEST_CHAR{'a'};
#ifdef Windows
const char* const TEST_PIPE_NAME{"\\\\.\\pipe\\testpipe"};
#else
const char* const TEST_PIPE_NAME{"testfiles/testpipe"};
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
            CPPUNIT_ASSERT(attempt++ <= MAX_ATTEMPTS);
            ml::core::CSleep::sleep(PAUSE_TIME_MS);
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

class CThreadBlockCanceller : public ml::core::CThread {
public:
    CThreadBlockCanceller(ml::core::CThread::TThreadId threadId)
        : m_ThreadId{threadId}, m_HasCancelledBlockingCall{false} {}

    const atomic_t::atomic_bool& hasCancelledBlockingCall() {
        return m_HasCancelledBlockingCall;
    }

protected:
    void run() override {
        // Wait for the file to exist
        ml::core::CSleep::sleep(SLEEP_TIME_MS);

        // Cancel the open() or read() operation on the file
        m_HasCancelledBlockingCall.store(true);
        CPPUNIT_ASSERT(ml::core::CThread::cancelBlockedIo(m_ThreadId));
    }

    void shutdown() override {}

private:
    ml::core::CThread::TThreadId m_ThreadId;
    atomic_t::atomic_bool m_HasCancelledBlockingCall;
};
}

CppUnit::Test* CNamedPipeFactoryTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CNamedPipeFactoryTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CNamedPipeFactoryTest>(
        "CNamedPipeFactoryTest::testServerIsCppReader",
        &CNamedPipeFactoryTest::testServerIsCppReader));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNamedPipeFactoryTest>(
        "CNamedPipeFactoryTest::testServerIsCReader", &CNamedPipeFactoryTest::testServerIsCReader));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNamedPipeFactoryTest>(
        "CNamedPipeFactoryTest::testServerIsCppWriter",
        &CNamedPipeFactoryTest::testServerIsCppWriter));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNamedPipeFactoryTest>(
        "CNamedPipeFactoryTest::testServerIsCWriter", &CNamedPipeFactoryTest::testServerIsCWriter));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNamedPipeFactoryTest>(
        "CNamedPipeFactoryTest::testCancelBlock", &CNamedPipeFactoryTest::testCancelBlock));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNamedPipeFactoryTest>(
        "CNamedPipeFactoryTest::testErrorIfRegularFile",
        &CNamedPipeFactoryTest::testErrorIfRegularFile));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNamedPipeFactoryTest>(
        "CNamedPipeFactoryTest::testErrorIfSymlink", &CNamedPipeFactoryTest::testErrorIfSymlink));

    return suiteOfTests;
}

void CNamedPipeFactoryTest::testServerIsCppReader() {
    CThreadDataWriter threadWriter{TEST_PIPE_NAME, TEST_SIZE};
    CPPUNIT_ASSERT(threadWriter.start());

    atomic_t::atomic_bool dummy{false};
    ml::core::CNamedPipeFactory::TIStreamP strm{
        ml::core::CNamedPipeFactory::openPipeStreamRead(TEST_PIPE_NAME, dummy)};
    CPPUNIT_ASSERT(strm);

    static const std::streamsize BUF_SIZE{512};
    std::string readData;
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

void CNamedPipeFactoryTest::testServerIsCReader() {
    CThreadDataWriter threadWriter{TEST_PIPE_NAME, TEST_SIZE};
    CPPUNIT_ASSERT(threadWriter.start());

    atomic_t::atomic_bool dummy{false};
    ml::core::CNamedPipeFactory::TFileP file{
        ml::core::CNamedPipeFactory::openPipeFileRead(TEST_PIPE_NAME, dummy)};
    CPPUNIT_ASSERT(file);

    static const std::size_t BUF_SIZE{512};
    std::string readData;
    char buffer[BUF_SIZE];
    do {
        std::size_t charsRead{std::fread(buffer, sizeof(char), BUF_SIZE, file.get())};
        CPPUNIT_ASSERT(!std::ferror(file.get()));
        if (charsRead > 0) {
            readData.append(buffer, charsRead);
        }
    } while (!std::feof(file.get()));

    CPPUNIT_ASSERT_EQUAL(TEST_SIZE, readData.length());
    CPPUNIT_ASSERT_EQUAL(std::string(TEST_SIZE, TEST_CHAR), readData);

    CPPUNIT_ASSERT(threadWriter.stop());

    file.reset();
}

void CNamedPipeFactoryTest::testServerIsCppWriter() {
    CThreadDataReader threadReader{TEST_PIPE_NAME};
    CPPUNIT_ASSERT(threadReader.start());

    atomic_t::atomic_bool dummy{false};
    ml::core::CNamedPipeFactory::TOStreamP strm{
        ml::core::CNamedPipeFactory::openPipeStreamWrite(TEST_PIPE_NAME, dummy)};
    CPPUNIT_ASSERT(strm);

    std::size_t charsLeft{TEST_SIZE};
    std::size_t blockSize{7};
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

void CNamedPipeFactoryTest::testServerIsCWriter() {
    CThreadDataReader threadReader{TEST_PIPE_NAME};
    CPPUNIT_ASSERT(threadReader.start());

    atomic_t::atomic_bool dummy{false};
    ml::core::CNamedPipeFactory::TFileP file{
        ml::core::CNamedPipeFactory::openPipeFileWrite(TEST_PIPE_NAME, dummy)};
    CPPUNIT_ASSERT(file);

    std::size_t charsLeft{TEST_SIZE};
    std::size_t blockSize{7};
    while (charsLeft > 0) {
        if (blockSize > charsLeft) {
            blockSize = charsLeft;
        }
        CPPUNIT_ASSERT(std::fputs(std::string(blockSize, TEST_CHAR).c_str(),
                                  file.get()) >= 0);
        charsLeft -= blockSize;
    }

    file.reset();

    CPPUNIT_ASSERT(threadReader.stop());

    CPPUNIT_ASSERT_EQUAL(TEST_SIZE, threadReader.data().length());
    CPPUNIT_ASSERT_EQUAL(std::string(TEST_SIZE, TEST_CHAR), threadReader.data());
}

void CNamedPipeFactoryTest::testCancelBlock() {
    CThreadBlockCanceller cancellerThread{ml::core::CThread::currentThreadId()};
    CPPUNIT_ASSERT(cancellerThread.start());

    ml::core::CNamedPipeFactory::TOStreamP strm{ml::core::CNamedPipeFactory::openPipeStreamWrite(
        TEST_PIPE_NAME, cancellerThread.hasCancelledBlockingCall())};
    CPPUNIT_ASSERT(strm == nullptr);

    CPPUNIT_ASSERT(cancellerThread.stop());
}

void CNamedPipeFactoryTest::testErrorIfRegularFile() {
    atomic_t::atomic_bool dummy{false};
    ml::core::CNamedPipeFactory::TIStreamP strm{
        ml::core::CNamedPipeFactory::openPipeStreamRead("Main.cc", dummy)};
    CPPUNIT_ASSERT(strm == nullptr);
}

void CNamedPipeFactoryTest::testErrorIfSymlink() {
#ifdef Windows
    // It's impossible to create a symlink to a named pipe on Windows - they
    // live under \\.\pipe\ and it's not possible to symlink to this part of
    // the file system
    LOG_DEBUG(<< "symlink test not relevant to Windows");
#else
    static const char* const TEST_SYMLINK_NAME{"test_symlink"};

    // Remove any files left behind by a previous failed test, but don't check
    // the return codes as these calls will usually fail
    ::unlink(TEST_SYMLINK_NAME);
    ::unlink(TEST_PIPE_NAME);

    CPPUNIT_ASSERT_EQUAL(0, ::mkfifo(TEST_PIPE_NAME, S_IRUSR | S_IWUSR));
    CPPUNIT_ASSERT_EQUAL(0, ::symlink(TEST_PIPE_NAME, TEST_SYMLINK_NAME));

    atomic_t::atomic_bool dummy{false};
    ml::core::CNamedPipeFactory::TIStreamP strm{
        ml::core::CNamedPipeFactory::openPipeStreamRead(TEST_SYMLINK_NAME, dummy)};
    CPPUNIT_ASSERT(strm == nullptr);

    CPPUNIT_ASSERT_EQUAL(0, ::unlink(TEST_SYMLINK_NAME));
    CPPUNIT_ASSERT_EQUAL(0, ::unlink(TEST_PIPE_NAME));
#endif
}
