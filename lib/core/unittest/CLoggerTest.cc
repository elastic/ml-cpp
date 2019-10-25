/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CNamedPipeFactory.h>
#include <core/COsFileFuncs.h>
#include <core/CSleep.h>

#include <rapidjson/document.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <fstream>
#include <ios>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

BOOST_AUTO_TEST_SUITE(CLoggerTest)

namespace {
#ifdef Windows
const char* const TEST_PIPE_NAME = "\\\\.\\pipe\\testpipe";
#else
const char* const TEST_PIPE_NAME = "testfiles/testpipe";
#endif
}

class CTestFixture {
public:
    ~CTestFixture() {
        // Tests in this file can leave the logger in an unusual state, so reset it
        // after each test
        ml::core::CLogger::instance().reset();
    }
};

BOOST_FIXTURE_TEST_CASE(testLogging, CTestFixture) {
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

        BOOST_TEST_REQUIRE(false);
    } catch (std::runtime_error&) { BOOST_TEST_REQUIRE(true); }
}

BOOST_FIXTURE_TEST_CASE(testReconfiguration, CTestFixture) {
    ml::core::CLogger& logger = ml::core::CLogger::instance();

    LOG_DEBUG(<< "Starting logger reconfiguration test");

    LOG_TRACE(<< "This shouldn't be seen because the hardcoded default log level is DEBUG");
    BOOST_TEST_REQUIRE(!logger.hasBeenReconfigured());

    BOOST_TEST_REQUIRE(!logger.reconfigureFromFile("nonexistantfile"));

    BOOST_TEST_REQUIRE(logger.reconfigureLogJson());
    LOG_INFO(<< "This should be logged as JSON!");

    // The test boost.log.ini is very similar to the hardcoded default, but
    // with the level set to TRACE rather than DEBUG
    BOOST_TEST_REQUIRE(logger.reconfigureFromFile("testfiles/boost.log.ini"));

    LOG_TRACE(<< "This should be seen because the reconfigured log level is TRACE");
    BOOST_TEST_REQUIRE(logger.hasBeenReconfigured());
}

BOOST_FIXTURE_TEST_CASE(testSetLevel, CTestFixture) {
    ml::core::CLogger& logger = ml::core::CLogger::instance();

    LOG_DEBUG(<< "Starting logger level test");

    BOOST_TEST_REQUIRE(logger.setLoggingLevel(ml::core::CLogger::E_Error));

    LOG_TRACE(<< "SHOULD NOT BE SEEN");
    LOG_DEBUG(<< "SHOULD NOT BE SEEN");
    LOG_INFO(<< "SHOULD NOT BE SEEN");
    LOG_WARN(<< "SHOULD NOT BE SEEN");
    LOG_ERROR(<< "Should be seen");
    LOG_FATAL(<< "Should be seen");

    BOOST_TEST_REQUIRE(logger.setLoggingLevel(ml::core::CLogger::E_Info));

    LOG_TRACE(<< "SHOULD NOT BE SEEN");
    LOG_DEBUG(<< "SHOULD NOT BE SEEN");
    LOG_INFO(<< "Should be seen");
    LOG_WARN(<< "Should be seen");
    LOG_ERROR(<< "Should be seen");
    LOG_FATAL(<< "Should be seen");

    BOOST_TEST_REQUIRE(logger.setLoggingLevel(ml::core::CLogger::E_Trace));

    LOG_TRACE(<< "Should be seen");
    LOG_DEBUG(<< "Should be seen");
    LOG_INFO(<< "Should be seen");
    LOG_WARN(<< "Should be seen");
    LOG_ERROR(<< "Should be seen");
    LOG_FATAL(<< "Should be seen");

    BOOST_TEST_REQUIRE(logger.setLoggingLevel(ml::core::CLogger::E_Warn));

    LOG_TRACE(<< "SHOULD NOT BE SEEN");
    LOG_DEBUG(<< "SHOULD NOT BE SEEN");
    LOG_INFO(<< "SHOULD NOT BE SEEN");
    LOG_WARN(<< "Should be seen");
    LOG_ERROR(<< "Should be seen");
    LOG_FATAL(<< "Should be seen");

    BOOST_TEST_REQUIRE(logger.setLoggingLevel(ml::core::CLogger::E_Fatal));

    LOG_TRACE(<< "SHOULD NOT BE SEEN");
    LOG_DEBUG(<< "SHOULD NOT BE SEEN");
    LOG_INFO(<< "SHOULD NOT BE SEEN");
    LOG_WARN(<< "SHOULD NOT BE SEEN");
    LOG_ERROR(<< "SHOULD NOT BE SEEN");
    LOG_FATAL(<< "Should be seen");

    BOOST_TEST_REQUIRE(logger.setLoggingLevel(ml::core::CLogger::E_Debug));

    LOG_DEBUG(<< "Finished logger level test");
}

BOOST_FIXTURE_TEST_CASE(testNonAsciiJsonLogging, CTestFixture) {
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
        BOOST_FAIL("Failed to connect to logging pipe within a reasonable time");
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
        BOOST_TEST_REQUIRE(!doc.HasParseError());
        BOOST_TEST_REQUIRE(doc.HasMember("message"));
        const rapidjson::Value& messageValue = doc["message"];
        std::string messageString(messageValue.GetString(), messageValue.GetStringLength());

        // we expect messages to be in order, so we only need to test the current one
        if (messageString.find(messages[foundMessages]) != std::string::npos) {
            ++foundMessages;
        } else if (foundMessages > 0) {
            BOOST_FAIL(messageString + " did not contain " + messages[foundMessages]);
        }
    }
    BOOST_REQUIRE_EQUAL(messages.size(), foundMessages);
}

BOOST_FIXTURE_TEST_CASE(testLogEnvironment, CTestFixture) {
    ml::core::CLogger::instance().logEnvironment();
}

BOOST_AUTO_TEST_SUITE_END()
