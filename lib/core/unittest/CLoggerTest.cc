/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CLoggerTest.h"

#include <core/CLogger.h>
#include <core/CNamedPipeFactory.h>
#include <core/COsFileFuncs.h>
#include <core/CSleep.h>

#include <rapidjson/document.h>

#include <algorithm>
#include <ios>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
#ifdef Windows
const char* const TEST_PIPE_NAME = "\\\\.\\pipe\\testpipe";
#else
const char* const TEST_PIPE_NAME = "testfiles/testpipe";
#endif
}

CppUnit::Test* CLoggerTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CLoggerTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CLoggerTest>(
        "CLoggerTest::testLogging", &CLoggerTest::testLogging));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLoggerTest>(
        "CLoggerTest::testReconfiguration", &CLoggerTest::testReconfiguration));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLoggerTest>(
        "CLoggerTest::testSetLevel", &CLoggerTest::testSetLevel));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLoggerTest>(
        "CLoggerTest::testLogEnvironment", &CLoggerTest::testLogEnvironment));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLoggerTest>(
        "CLoggerTest::testNonAsciiJsonLogging", &CLoggerTest::testNonAsciiJsonLogging));

    return suiteOfTests;
}

void CLoggerTest::tearDown() {
    // Tests in this file can leave the logger in an unusual state, so reset it
    // after each test
    ml::core::CLogger::instance().reset();
}

void CLoggerTest::testLogging() {
    std::string t("Test message");

    LOG_TRACE(<< "Trace");
    LOG_AT_LEVEL(ml::core::CLogger::E_Trace, << "Dynamic TRACE " << 1);
    LOG_DEBUG(<< "Debug");
    LOG_AT_LEVEL(ml::core::CLogger::E_Debug, << "Dynamic DEBUG " << 2.0);
    LOG_INFO(<< "Info " << std::boolalpha << true);
    LOG_AT_LEVEL(ml::core::CLogger::E_Info, << "Dynamic INFO " << false);
    LOG_WARN(<< "Warn " << t);
    LOG_AT_LEVEL(ml::core::CLogger::E_Warn, << "Dynamic WARN "
                                            << "abc");
    LOG_ERROR(<< "Error " << 1000 << ' ' << 0.23124F);
    LOG_AT_LEVEL(ml::core::CLogger::E_Error, << "Dynamic ERROR");
    LOG_FATAL(<< "Fatal - application to handle exit");
    LOG_AT_LEVEL(ml::core::CLogger::E_Fatal, << "Dynamic FATAL " << t);
    try {
        LOG_ABORT(<< "Throwing exception " << 1221U << ' ' << 0.23124);

        CPPUNIT_ASSERT(false);
    } catch (std::runtime_error&) { CPPUNIT_ASSERT(true); }
}

void CLoggerTest::testReconfiguration() {
    ml::core::CLogger& logger = ml::core::CLogger::instance();

    LOG_DEBUG(<< "Starting logger reconfiguration test");

    LOG_TRACE(<< "This shouldn't be seen because the hardcoded default log level is DEBUG");
    CPPUNIT_ASSERT(!logger.hasBeenReconfigured());

    CPPUNIT_ASSERT(!logger.reconfigureFromFile("nonexistantfile"));

    CPPUNIT_ASSERT(logger.reconfigureLogJson());
    LOG_INFO(<< "This should be logged as JSON!");

    // The test boost.log.ini is very similar to the hardcoded default, but
    // with the level set to TRACE rather than DEBUG
    CPPUNIT_ASSERT(logger.reconfigureFromFile("testfiles/boost.log.ini"));

    LOG_TRACE(<< "This should be seen because the reconfigured log level is TRACE");
    CPPUNIT_ASSERT(logger.hasBeenReconfigured());
}

void CLoggerTest::testSetLevel() {
    ml::core::CLogger& logger = ml::core::CLogger::instance();

    LOG_DEBUG(<< "Starting logger level test");

    CPPUNIT_ASSERT(logger.setLoggingLevel(ml::core::CLogger::E_Error));

    LOG_TRACE(<< "SHOULD NOT BE SEEN");
    LOG_DEBUG(<< "SHOULD NOT BE SEEN");
    LOG_INFO(<< "SHOULD NOT BE SEEN");
    LOG_WARN(<< "SHOULD NOT BE SEEN");
    LOG_ERROR(<< "Should be seen");
    LOG_FATAL(<< "Should be seen");

    CPPUNIT_ASSERT(logger.setLoggingLevel(ml::core::CLogger::E_Info));

    LOG_TRACE(<< "SHOULD NOT BE SEEN");
    LOG_DEBUG(<< "SHOULD NOT BE SEEN");
    LOG_INFO(<< "Should be seen");
    LOG_WARN(<< "Should be seen");
    LOG_ERROR(<< "Should be seen");
    LOG_FATAL(<< "Should be seen");

    CPPUNIT_ASSERT(logger.setLoggingLevel(ml::core::CLogger::E_Trace));

    LOG_TRACE(<< "Should be seen");
    LOG_DEBUG(<< "Should be seen");
    LOG_INFO(<< "Should be seen");
    LOG_WARN(<< "Should be seen");
    LOG_ERROR(<< "Should be seen");
    LOG_FATAL(<< "Should be seen");

    CPPUNIT_ASSERT(logger.setLoggingLevel(ml::core::CLogger::E_Warn));

    LOG_TRACE(<< "SHOULD NOT BE SEEN");
    LOG_DEBUG(<< "SHOULD NOT BE SEEN");
    LOG_INFO(<< "SHOULD NOT BE SEEN");
    LOG_WARN(<< "Should be seen");
    LOG_ERROR(<< "Should be seen");
    LOG_FATAL(<< "Should be seen");

    CPPUNIT_ASSERT(logger.setLoggingLevel(ml::core::CLogger::E_Fatal));

    LOG_TRACE(<< "SHOULD NOT BE SEEN");
    LOG_DEBUG(<< "SHOULD NOT BE SEEN");
    LOG_INFO(<< "SHOULD NOT BE SEEN");
    LOG_WARN(<< "SHOULD NOT BE SEEN");
    LOG_ERROR(<< "SHOULD NOT BE SEEN");
    LOG_FATAL(<< "Should be seen");

    CPPUNIT_ASSERT(logger.setLoggingLevel(ml::core::CLogger::E_Debug));

    LOG_DEBUG(<< "Finished logger level test");
}

void CLoggerTest::testNonAsciiJsonLogging() {
    std::vector<std::string> messages{"Non-iso8859-15: 编码", "Non-ascii: üaöä",
                                      "Non-iso8859-15: 编码 test", "surrogate pair: 𐐷 test"};

    std::ostringstream loggedData;
    std::thread reader([&loggedData] {
        for (std::size_t attempt = 1; attempt <= 100; ++attempt) {
            // wait a bit so that pipe has been created
            ml::core::CSleep::sleep(50);
            std::ifstream strm(TEST_PIPE_NAME);
            if (strm.is_open()) {
                std::copy(std::istreambuf_iterator<char>(strm),
                          std::istreambuf_iterator<char>(),
                          std::ostreambuf_iterator<char>(loggedData));
                return;
            }
        }
        CPPUNIT_FAIL("Failed to connect to logging pipe within a reasonable time");
    });

    ml::core::CLogger& logger = ml::core::CLogger::instance();
    // logger might got reconfigured in previous tests, so reset and reconfigure it
    logger.reset();
    logger.reconfigure(TEST_PIPE_NAME, "");

    for (const auto& m : messages) {
        LOG_INFO(<< m);
    }

    // reset the logger to end the stream and revert state for following tests
    logger.reset();

    reader.join();
    std::istringstream inputStream(loggedData.str());
    std::string line;
    size_t foundMessages = 0;

    // test that we found the messages we put in,
    while (std::getline(inputStream, line)) {
        if (line.empty()) {
            continue;
        }
        rapidjson::Document doc;
        doc.Parse<rapidjson::kParseDefaultFlags>(line);
        CPPUNIT_ASSERT(!doc.HasParseError());
        CPPUNIT_ASSERT(doc.HasMember("message"));
        const rapidjson::Value& messageValue = doc["message"];
        std::string messageString(messageValue.GetString(), messageValue.GetStringLength());

        // we expect messages to be in order, so we only need to test the current one
        if (messageString.find(messages[foundMessages]) != std::string::npos) {
            ++foundMessages;
        } else if (foundMessages > 0) {
            CPPUNIT_FAIL(messageString + " did not contain " + messages[foundMessages]);
        }
    }
    CPPUNIT_ASSERT_EQUAL(messages.size(), foundMessages);
}

void CLoggerTest::testLogEnvironment() {
    ml::core::CLogger::instance().logEnvironment();
}
