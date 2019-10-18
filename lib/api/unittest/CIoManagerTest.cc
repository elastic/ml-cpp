/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CMutex.h>
#include <core/CNamedPipeFactory.h>
#include <core/CScopedLock.h>
#include <core/CSleep.h>
#include <core/CThread.h>

#include <api/CIoManager.h>

#include <boost/test/unit_test.hpp>

#include <fstream>
#include <iostream>

#include <stdio.h>

BOOST_AUTO_TEST_SUITE(CIoManagerTest)

namespace {

const uint32_t SLEEP_TIME_MS = 100;
const uint32_t PAUSE_TIME_MS = 10;
const size_t MAX_ATTEMPTS = 100;
const size_t TEST_SIZE = 10000;
const char TEST_CHAR = 'a';
const char* const GOOD_INPUT_FILE_NAME = "testfiles/good_input_file";
const char* const GOOD_OUTPUT_FILE_NAME = "testfiles/good_output_file";
#ifdef Windows
const char* const GOOD_INPUT_PIPE_NAME = "\\\\.\\pipe\\good_input_pipe";
const char* const GOOD_OUTPUT_PIPE_NAME = "\\\\.\\pipe\\good_output_pipe";
#else
const char* const GOOD_INPUT_PIPE_NAME = "testfiles/good_input_pipe";
const char* const GOOD_OUTPUT_PIPE_NAME = "testfiles/good_output_pipe";
#endif
const char* const BAD_INPUT_FILE_NAME = "can't_create_a_file_here/bad_input_file";
const char* const BAD_OUTPUT_FILE_NAME = "can't_create_a_file_here/bad_output_file";
const char* const BAD_INPUT_PIPE_NAME = "can't_create_a_pipe_here/bad_input_pipe";
const char* const BAD_OUTPUT_PIPE_NAME = "can't_create_a_pipe_here/bad_output_pipe";

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
    CThreadDataReader(const std::string& fileName)
        : m_FileName(fileName), m_Shutdown(false) {}

    const std::string& data() const {
        // The memory barriers associated with the mutex lock should ensure
        // the thread calling this method has as up-to-date view of m_Data's
        // member variables as the thread that updated it
        ml::core::CScopedLock lock(m_Mutex);
        return m_Data;
    }

protected:
    virtual void run() {
        m_Data.clear();

        std::ifstream strm;

        // Try to open the file repeatedly to allow time for the other
        // thread to create it
        size_t attempt(1);
        do {
            if (m_Shutdown) {
                return;
            }
            BOOST_TEST(attempt++ <= MAX_ATTEMPTS);
            ml::core::CSleep::sleep(PAUSE_TIME_MS);
            strm.open(m_FileName.c_str());
        } while (!strm.is_open());

        static const std::streamsize BUF_SIZE = 512;
        char buffer[BUF_SIZE];
        while (strm.good()) {
            if (m_Shutdown) {
                return;
            }
            strm.read(buffer, BUF_SIZE);
            BOOST_TEST(!strm.bad());
            if (strm.gcount() > 0) {
                ml::core::CScopedLock lock(m_Mutex);
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

    virtual void shutdown() { m_Shutdown = true; }

private:
    mutable ml::core::CMutex m_Mutex;
    std::string m_FileName;
    std::string m_Data;
    volatile bool m_Shutdown;
};
}


BOOST_AUTO_TEST_CASE(testStdinStdout) {
    ml::api::CIoManager ioMgr("", false, "", false);
    BOOST_TEST(ioMgr.initIo());

    // Assign to a different pointer in case of "this" pointer manipulation due
    // to multiple inheritance
    std::istream* cinAsIStream = &std::cin;
    BOOST_CHECK_EQUAL(cinAsIStream, &ioMgr.inputStream());

    std::ostream* coutAsIStream = &std::cout;
    BOOST_CHECK_EQUAL(coutAsIStream, &ioMgr.outputStream());
}

BOOST_AUTO_TEST_CASE(testFileIoGood) {
    // Remove output file that possibly might have been left behind by a
    // previous failed test - ignore the error code from this call though as
    // it'll generally fail
    ::remove(GOOD_OUTPUT_FILE_NAME);

    // For the file test the input file needs to exist before the IO manager
    // is started
    std::ofstream strm(GOOD_INPUT_FILE_NAME);
    strm << std::string(TEST_SIZE, TEST_CHAR);
    strm.close();

    this->testCommon(GOOD_INPUT_FILE_NAME, false, GOOD_OUTPUT_FILE_NAME, false, true);

    BOOST_CHECK_EQUAL(0, ::remove(GOOD_INPUT_FILE_NAME));
    BOOST_CHECK_EQUAL(0, ::remove(GOOD_OUTPUT_FILE_NAME));
}

BOOST_AUTO_TEST_CASE(testFileIoBad) {
    this->testCommon(BAD_INPUT_FILE_NAME, false, BAD_OUTPUT_FILE_NAME, false, false);
}

BOOST_AUTO_TEST_CASE(testNamedPipeIoGood) {
    // For the named pipe test, data needs to be written to the IO manager's
    // input pipe after the IO manager has started
    CThreadDataWriter threadWriter(GOOD_INPUT_PIPE_NAME, TEST_SIZE);
    BOOST_TEST(threadWriter.start());

    this->testCommon(GOOD_INPUT_PIPE_NAME, true, GOOD_OUTPUT_PIPE_NAME, true, true);

    BOOST_TEST(threadWriter.stop());
}

BOOST_AUTO_TEST_CASE(testNamedPipeIoBad) {
    this->testCommon(BAD_INPUT_PIPE_NAME, true, BAD_OUTPUT_PIPE_NAME, true, false);
}

BOOST_AUTO_TEST_CASE(testCommonconst std::string& inputFileName,
                                bool isInputFileNamedPipe,
                                const std::string& outputFileName,
                                bool isOutputFileNamedPipe,
                                bool isGood) {
    // Test reader reads from the IO manager's output stream.
    CThreadDataReader threadReader(outputFileName);
    BOOST_TEST(threadReader.start());

    std::string processedData;

    {
        ml::api::CIoManager ioMgr(inputFileName, isInputFileNamedPipe,
                                  outputFileName, isOutputFileNamedPipe);
        BOOST_CHECK_EQUAL(isGood, ioMgr.initIo());
        if (isGood) {
            static const std::streamsize BUF_SIZE = 512;
            char buffer[BUF_SIZE];
            do {
                ioMgr.inputStream().read(buffer, BUF_SIZE);
                BOOST_TEST(!ioMgr.inputStream().bad());
                if (ioMgr.inputStream().gcount() > 0) {
                    processedData.append(
                        buffer, static_cast<size_t>(ioMgr.inputStream().gcount()));
                }
                BOOST_TEST(!ioMgr.outputStream().bad());
                ioMgr.outputStream().write(
                    buffer, static_cast<size_t>(ioMgr.inputStream().gcount()));
            } while (!ioMgr.inputStream().eof());
            BOOST_TEST(!ioMgr.outputStream().bad());
        }
    }

    if (isGood) {
        BOOST_TEST(threadReader.waitForFinish());
        BOOST_CHECK_EQUAL(TEST_SIZE, processedData.length());
        BOOST_CHECK_EQUAL(std::string(TEST_SIZE, TEST_CHAR), processedData);
        BOOST_CHECK_EQUAL(processedData.length(), threadReader.data().length());
        BOOST_CHECK_EQUAL(processedData, threadReader.data());
    } else {
        BOOST_TEST(threadReader.stop());
        BOOST_TEST(processedData.empty());
    }
}

BOOST_AUTO_TEST_SUITE_END()
