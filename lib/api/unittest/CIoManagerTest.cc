/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CIoManager.h>

#include <test/CThreadDataReader.h>
#include <test/CThreadDataWriter.h>

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

void testCommon(const std::string& inputFileName,
                bool isInputFileNamedPipe,
                const std::string& outputFileName,
                bool isOutputFileNamedPipe,
                bool isGood) {
    // Test reader reads from the IO manager's output stream.
    ml::test::CThreadDataReader threadReader(PAUSE_TIME_MS, MAX_ATTEMPTS, outputFileName);
    BOOST_TEST_REQUIRE(threadReader.start());

    std::string processedData;

    {
        ml::api::CIoManager ioMgr(inputFileName, isInputFileNamedPipe,
                                  outputFileName, isOutputFileNamedPipe);
        BOOST_REQUIRE_EQUAL(isGood, ioMgr.initIo());
        if (isGood) {
            static const std::streamsize BUF_SIZE = 512;
            char buffer[BUF_SIZE];
            do {
                ioMgr.inputStream().read(buffer, BUF_SIZE);
                BOOST_TEST_REQUIRE(!ioMgr.inputStream().bad());
                if (ioMgr.inputStream().gcount() > 0) {
                    processedData.append(
                        buffer, static_cast<size_t>(ioMgr.inputStream().gcount()));
                }
                BOOST_TEST_REQUIRE(!ioMgr.outputStream().bad());
                ioMgr.outputStream().write(
                    buffer, static_cast<size_t>(ioMgr.inputStream().gcount()));
            } while (!ioMgr.inputStream().eof());
            BOOST_TEST_REQUIRE(!ioMgr.outputStream().bad());
        }
    }

    if (isGood) {
        BOOST_TEST_REQUIRE(threadReader.waitForFinish());
        BOOST_TEST_REQUIRE(threadReader.attemptsTaken() <= MAX_ATTEMPTS);
        BOOST_TEST_REQUIRE(threadReader.streamWentBad() == false);
        BOOST_REQUIRE_EQUAL(TEST_SIZE, processedData.length());
        BOOST_REQUIRE_EQUAL(std::string(TEST_SIZE, TEST_CHAR), processedData);
        BOOST_REQUIRE_EQUAL(processedData.length(), threadReader.data().length());
        BOOST_REQUIRE_EQUAL(processedData, threadReader.data());
    } else {
        BOOST_TEST_REQUIRE(threadReader.stop());
        BOOST_TEST_REQUIRE(processedData.empty());
    }
}
}

BOOST_AUTO_TEST_CASE(testStdinStdout) {
    ml::api::CIoManager ioMgr("", false, "", false);
    BOOST_TEST_REQUIRE(ioMgr.initIo());

    // Assign to a different pointer in case of "this" pointer manipulation due
    // to multiple inheritance
    std::istream* cinAsIStream = &std::cin;
    BOOST_REQUIRE_EQUAL(cinAsIStream, &ioMgr.inputStream());

    std::ostream* coutAsOStream = &std::cout;
    BOOST_REQUIRE_EQUAL(coutAsOStream, &ioMgr.outputStream());
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

    testCommon(GOOD_INPUT_FILE_NAME, false, GOOD_OUTPUT_FILE_NAME, false, true);

    BOOST_REQUIRE_EQUAL(0, ::remove(GOOD_INPUT_FILE_NAME));
    BOOST_REQUIRE_EQUAL(0, ::remove(GOOD_OUTPUT_FILE_NAME));
}

BOOST_AUTO_TEST_CASE(testFileIoBad) {
    testCommon(BAD_INPUT_FILE_NAME, false, BAD_OUTPUT_FILE_NAME, false, false);
}

BOOST_AUTO_TEST_CASE(testNamedPipeIoGood) {
    // For the named pipe test, data needs to be written to the IO manager's
    // input pipe after the IO manager has started
    ml::test::CThreadDataWriter threadWriter(SLEEP_TIME_MS, GOOD_INPUT_PIPE_NAME,
                                             TEST_CHAR, TEST_SIZE);
    BOOST_TEST_REQUIRE(threadWriter.start());

    testCommon(GOOD_INPUT_PIPE_NAME, true, GOOD_OUTPUT_PIPE_NAME, true, true);

    BOOST_TEST_REQUIRE(threadWriter.waitForFinish());
}

BOOST_AUTO_TEST_CASE(testNamedPipeIoBad) {
    testCommon(BAD_INPUT_PIPE_NAME, true, BAD_OUTPUT_PIPE_NAME, true, false);
}

BOOST_AUTO_TEST_SUITE_END()
