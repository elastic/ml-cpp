/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CIoManagerTest.h"

#include <core/CMutex.h>
#include <core/CNamedPipeFactory.h>
#include <core/CScopedLock.h>
#include <core/CSleep.h>
#include <core/CThread.h>

#include <api/CIoManager.h>

#include <fstream>
#include <iostream>

#include <stdio.h>


namespace
{

const uint32_t SLEEP_TIME_MS = 100;
const uint32_t PAUSE_TIME_MS = 10;
const size_t   MAX_ATTEMPTS = 100;
const size_t   TEST_SIZE = 10000;
const char     TEST_CHAR = 'a';
const char     *GOOD_INPUT_FILE_NAME = "testfiles/good_input_file";
const char     *GOOD_OUTPUT_FILE_NAME = "testfiles/good_output_file";
#ifdef Windows
const char     *GOOD_INPUT_PIPE_NAME = "\\\\.\\pipe\\good_input_pipe";
const char     *GOOD_OUTPUT_PIPE_NAME = "\\\\.\\pipe\\good_output_pipe";
#else
const char     *GOOD_INPUT_PIPE_NAME = "testfiles/good_input_pipe";
const char     *GOOD_OUTPUT_PIPE_NAME = "testfiles/good_output_pipe";
#endif
const char     *BAD_INPUT_FILE_NAME = "can't_create_a_file_here/bad_input_file";
const char     *BAD_OUTPUT_FILE_NAME = "can't_create_a_file_here/bad_output_file";
const char     *BAD_INPUT_PIPE_NAME = "can't_create_a_pipe_here/bad_input_pipe";
const char     *BAD_OUTPUT_PIPE_NAME = "can't_create_a_pipe_here/bad_output_pipe";

class CThreadDataWriter : public ml::core::CThread
{
    public:
        CThreadDataWriter(const std::string &fileName, size_t size)
            : m_FileName(fileName),
              m_Size(size)
        {
        }

    protected:
        virtual void run(void)
        {
            // Wait for the file to exist
            ml::core::CSleep::sleep(SLEEP_TIME_MS);

            std::ofstream strm(m_FileName.c_str());
            for (size_t i = 0; i < m_Size && strm.good(); ++i)
            {
                strm << TEST_CHAR;
            }
        }

        virtual void shutdown(void)
        {
        }

    private:
        std::string m_FileName;
        size_t      m_Size;
};

class CThreadDataReader : public ml::core::CThread
{
    public:
        CThreadDataReader(const std::string &fileName)
            : m_FileName(fileName),
              m_Shutdown(false)
        {
        }

        const std::string &data(void) const
        {
            // The memory barriers associated with the mutex lock should ensure
            // the thread calling this method has as up-to-date view of m_Data's
            // member variables as the thread that updated it
            ml::core::CScopedLock lock(m_Mutex);
            return m_Data;
        }

    protected:
        virtual void run(void)
        {
            m_Data.clear();

            std::ifstream strm;

            // Try to open the file repeatedly to allow time for the other
            // thread to create it
            size_t attempt(1);
            do
            {
                if (m_Shutdown)
                {
                    return;
                }
                CPPUNIT_ASSERT(attempt++ <= MAX_ATTEMPTS);
                ml::core::CSleep::sleep(PAUSE_TIME_MS);
                strm.open(m_FileName.c_str());
            }
            while (!strm.is_open());

            static const std::streamsize BUF_SIZE = 512;
            char buffer[BUF_SIZE];
            while (strm.good())
            {
                if (m_Shutdown)
                {
                    return;
                }
                strm.read(buffer, BUF_SIZE);
                CPPUNIT_ASSERT(!strm.bad());
                if (strm.gcount() > 0)
                {
                    ml::core::CScopedLock lock(m_Mutex);
                    // This code deals with the test character we write to
                    // detect the short-lived connection problem on Windows
                    const char *copyFrom = buffer;
                    size_t copyLen = static_cast<size_t>(strm.gcount());
                    if (m_Data.empty() &&
                        *buffer == ml::core::CNamedPipeFactory::TEST_CHAR)
                    {
                        ++copyFrom;
                        --copyLen;
                    }
                    if (copyLen > 0)
                    {
                        m_Data.append(copyFrom, copyLen);
                    }
                }
            }
        }

        virtual void shutdown(void)
        {
            m_Shutdown = true;
        }

    private:
        mutable ml::core::CMutex m_Mutex;
        std::string                   m_FileName;
        std::string                   m_Data;
        volatile bool                 m_Shutdown;
};

}

CppUnit::Test *CIoManagerTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CIoManagerTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CIoManagerTest>(
                                   "CIoManagerTest::testStdinStdout",
                                   &CIoManagerTest::testStdinStdout) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CIoManagerTest>(
                                   "CIoManagerTest::testFileIoGood",
                                   &CIoManagerTest::testFileIoGood) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CIoManagerTest>(
                                   "CIoManagerTest::testFileIoBad",
                                   &CIoManagerTest::testFileIoBad) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CIoManagerTest>(
                                   "CIoManagerTest::testNamedPipeIoGood",
                                   &CIoManagerTest::testNamedPipeIoGood) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CIoManagerTest>(
                                   "CIoManagerTest::testNamedPipeIoBad",
                                   &CIoManagerTest::testNamedPipeIoBad) );

    return suiteOfTests;
}

void CIoManagerTest::testStdinStdout(void)
{
    ml::api::CIoManager ioMgr("", false, "", false);
    CPPUNIT_ASSERT(ioMgr.initIo());

    // Assign to a different pointer in case of "this" pointer manipulation due
    // to multiple inheritance
    std::istream *cinAsIStream = &std::cin;
    CPPUNIT_ASSERT_EQUAL(cinAsIStream, &ioMgr.inputStream());

    std::ostream *coutAsIStream = &std::cout;
    CPPUNIT_ASSERT_EQUAL(coutAsIStream, &ioMgr.outputStream());
}

void CIoManagerTest::testFileIoGood(void)
{
    // Remove output file that possibly might have been left behind by a
    // previous failed test - ignore the error code from this call though as
    // it'll generally fail
    ::remove(GOOD_OUTPUT_FILE_NAME);

    // For the file test the input file needs to exist before the IO manager
    // is started
    std::ofstream strm(GOOD_INPUT_FILE_NAME);
    strm << std::string(TEST_SIZE, TEST_CHAR);
    strm.close();

    this->testCommon(GOOD_INPUT_FILE_NAME,
                     false,
                     GOOD_OUTPUT_FILE_NAME,
                     false,
                     true);

    CPPUNIT_ASSERT_EQUAL(0, ::remove(GOOD_INPUT_FILE_NAME));
    CPPUNIT_ASSERT_EQUAL(0, ::remove(GOOD_OUTPUT_FILE_NAME));
}

void CIoManagerTest::testFileIoBad(void)
{
    this->testCommon(BAD_INPUT_FILE_NAME,
                     false,
                     BAD_OUTPUT_FILE_NAME,
                     false,
                     false);
}

void CIoManagerTest::testNamedPipeIoGood(void)
{
    // For the named pipe test, data needs to be written to the IO manager's
    // input pipe after the IO manager has started
    CThreadDataWriter threadWriter(GOOD_INPUT_PIPE_NAME, TEST_SIZE);
    CPPUNIT_ASSERT(threadWriter.start());

    this->testCommon(GOOD_INPUT_PIPE_NAME,
                     true,
                     GOOD_OUTPUT_PIPE_NAME,
                     true,
                     true);

    CPPUNIT_ASSERT(threadWriter.stop());
}

void CIoManagerTest::testNamedPipeIoBad(void)
{
    this->testCommon(BAD_INPUT_PIPE_NAME,
                     true,
                     BAD_OUTPUT_PIPE_NAME,
                     true,
                     false);
}

void CIoManagerTest::testCommon(const std::string &inputFileName,
                                bool isInputFileNamedPipe,
                                const std::string &outputFileName,
                                bool isOutputFileNamedPipe,
                                bool isGood)
{
    // Test reader reads from the IO manager's output stream.
    CThreadDataReader threadReader(outputFileName);
    CPPUNIT_ASSERT(threadReader.start());

    std::string processedData;

    {
        ml::api::CIoManager ioMgr(inputFileName,
                                       isInputFileNamedPipe,
                                       outputFileName,
                                       isOutputFileNamedPipe);
        CPPUNIT_ASSERT_EQUAL(isGood, ioMgr.initIo());
        if (isGood)
        {
            static const std::streamsize BUF_SIZE = 512;
            char buffer[BUF_SIZE];
            do
            {
                ioMgr.inputStream().read(buffer, BUF_SIZE);
                CPPUNIT_ASSERT(!ioMgr.inputStream().bad());
                if (ioMgr.inputStream().gcount() > 0)
                {
                    processedData.append(buffer, static_cast<size_t>(ioMgr.inputStream().gcount()));
                }
                CPPUNIT_ASSERT(!ioMgr.outputStream().bad());
                ioMgr.outputStream().write(buffer, static_cast<size_t>(ioMgr.inputStream().gcount()));
            }
            while (!ioMgr.inputStream().eof());
            CPPUNIT_ASSERT(!ioMgr.outputStream().bad());
        }
    }

    if (isGood)
    {
        CPPUNIT_ASSERT(threadReader.waitForFinish());
        CPPUNIT_ASSERT_EQUAL(TEST_SIZE, processedData.length());
        CPPUNIT_ASSERT_EQUAL(std::string(TEST_SIZE, TEST_CHAR), processedData);
        CPPUNIT_ASSERT_EQUAL(processedData.length(), threadReader.data().length());
        CPPUNIT_ASSERT_EQUAL(processedData, threadReader.data());
    }
    else
    {
        CPPUNIT_ASSERT(threadReader.stop());
        CPPUNIT_ASSERT(processedData.empty());
    }
}

