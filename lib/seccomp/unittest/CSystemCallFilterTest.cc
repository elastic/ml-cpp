/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/COsFileFuncs.h>
#include <core/CSleep.h>
#include <core/CThread.h>
#ifdef Linux
#include <core/CRegex.h>
#include <core/CUname.h>
#endif

#include <seccomp/CSystemCallFilter.h>

#include <test/CTestTmpDir.h>

#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/system/error_code.hpp>

#include <cstdlib>
#include <string>

BOOST_AUTO_TEST_SUITE(CSystemCallFilterTest)

namespace {

const uint32_t SLEEP_TIME_MS = 100;
const size_t TEST_SIZE = 10000;
const size_t MAX_ATTEMPTS = 20;
const char TEST_CHAR = 'a';
// CTestTmpDir::tmpDir() fails to get the current user after the system call
// filter is installed, so cache the value early
const std::string TMP_DIR{ml::test::CTestTmpDir::tmpDir()};
#ifdef Windows
const std::string TEST_READ_PIPE_NAME{"\\\\.\\pipe\\testreadpipe"};
const std::string TEST_WRITE_PIPE_NAME{"\\\\.\\pipe\\testwritepipe"};
#else
const std::string TEST_READ_PIPE_NAME{TMP_DIR + "/testreadpipe"};
const std::string TEST_WRITE_PIPE_NAME{TMP_DIR + "/testwritepipe"};
#endif

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
            BOOST_TEST(attempt++ <= MAX_ATTEMPTS);
            ml::core::CSleep::sleep(SLEEP_TIME_MS);
            strm.open(m_FileName.c_str());
        } while (!strm.is_open());

        static const std::streamsize BUF_SIZE = 512;
        char buffer[BUF_SIZE];
        while (strm.good()) {
            strm.read(buffer, BUF_SIZE);
            BOOST_TEST(!strm.bad());
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

bool systemCall() {
    return std::system("hostname") == 0;
}

#ifdef Linux
bool versionIsBefore3_5(std::int64_t major, std::int64_t minor) {
    if (major < 3) {
        return true;
    }
    if (major == 3 && minor < 5) {
        return true;
    }
    return false;
}
#endif
}


BOOST_AUTO_TEST_CASE(testSystemCallFilter) {
#ifdef Linux
    std::string release{ml::core::CUname::release()};
    ml::core::CRegex semVersion;
    BOOST_TEST(semVersion.init("(\\d)\\.(\\d{1,2})\\.(\\d{1,2}).*"));
    ml::core::CRegex::TStrVec tokens;
    BOOST_TEST(semVersion.tokenise(release, tokens));
    // Seccomp is available in kernels since 3.5

    std::int64_t major = std::stoi(tokens[0]);
    std::int64_t minor = std::stoi(tokens[1]);
    if (versionIsBefore3_5(major, minor)) {
        LOG_INFO(<< "Cannot test seccomp on linux kernels before 3.5");
        return;
    }
#endif // Linux

    // Ensure actions are not prohibited before the
    // system call filters are applied
    BOOST_TEST(systemCall());

    // Install the filter
    ml::seccomp::CSystemCallFilter::installSystemCallFilter();

    CPPUNIT_ASSERT_ASSERTION_FAIL_MESSAGE("Calling std::system should fail",
                                          BOOST_TEST(systemCall()));

    // Operations that must function after seccomp is initialised
    openPipeAndRead(TEST_READ_PIPE_NAME);
    openPipeAndWrite(TEST_WRITE_PIPE_NAME);

    makeAndRemoveDirectory(TMP_DIR);
}

void CSystemCallFilterTest::openPipeAndRead(const std::string& filename) {

    CNamedPipeWriter threadWriter(filename, TEST_SIZE);
    BOOST_TEST(threadWriter.start());

    ml::core::CNamedPipeFactory::TIStreamP strm =
        ml::core::CNamedPipeFactory::openPipeStreamRead(filename);
    BOOST_TEST(strm);

    static const std::streamsize BUF_SIZE = 512;
    std::string readData;
    readData.reserve(TEST_SIZE);
    char buffer[BUF_SIZE];
    do {
        strm->read(buffer, BUF_SIZE);
        BOOST_TEST(!strm->bad());
        if (strm->gcount() > 0) {
            readData.append(buffer, static_cast<size_t>(strm->gcount()));
        }
    } while (!strm->eof());

    BOOST_CHECK_EQUAL(TEST_SIZE, readData.length());
    BOOST_CHECK_EQUAL(std::string(TEST_SIZE, TEST_CHAR), readData);

    BOOST_TEST(threadWriter.stop());

    strm.reset();
}

void CSystemCallFilterTest::openPipeAndWrite(const std::string& filename) {
    CNamedPipeReader threadReader(filename);
    BOOST_TEST(threadReader.start());

    ml::core::CNamedPipeFactory::TOStreamP strm =
        ml::core::CNamedPipeFactory::openPipeStreamWrite(filename);
    BOOST_TEST(strm);

    size_t charsLeft(TEST_SIZE);
    size_t blockSize(7);
    while (charsLeft > 0) {
        if (blockSize > charsLeft) {
            blockSize = charsLeft;
        }
        (*strm) << std::string(blockSize, TEST_CHAR);
        BOOST_TEST(!strm->bad());
        charsLeft -= blockSize;
    }

    strm.reset();

    BOOST_TEST(threadReader.stop());

    BOOST_CHECK_EQUAL(TEST_SIZE, threadReader.data().length());
    BOOST_CHECK_EQUAL(std::string(TEST_SIZE, TEST_CHAR), threadReader.data());
}

void CSystemCallFilterTest::makeAndRemoveDirectory(const std::string& dirname) {

    boost::filesystem::path temporaryFolder(dirname);
    temporaryFolder /= "test-directory";

    boost::system::error_code errorCode;
    boost::filesystem::create_directories(temporaryFolder, errorCode);
    BOOST_CHECK_EQUAL(boost::system::error_code(), errorCode);
    boost::filesystem::remove_all(temporaryFolder, errorCode);
    BOOST_CHECK_EQUAL(boost::system::error_code(), errorCode);
}

BOOST_AUTO_TEST_SUITE_END()
