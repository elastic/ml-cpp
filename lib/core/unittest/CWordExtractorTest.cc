/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CWordExtractor.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CWordExtractorTest)

BOOST_AUTO_TEST_CASE(testWordExtract) {
    {
        std::string message("2017-01-25 02:10:03,551 ERROR [co.elastic.tradefeedtracker.MessageLoggerService] Failed to Rollback");
        std::string words;

        ml::core::CWordExtractor::extractWordsFromMessage(message, words);

        LOG_DEBUG(<< "Message: " << message);
        LOG_DEBUG(<< "Words: " << words);

        BOOST_CHECK_EQUAL(std::string("Failed to Rollback"), words);
    }
    {
        std::string message(
            "2017-01-25 14:20:49,646 INFO  [co.elastic.settlement.synchronization.errors.NonFXInstructionSyncImpl] Found "
            "corresponding outgoingPaymentFlow :: OutGoingPaymentFlow.id = 7480");
        std::string words;

        ml::core::CWordExtractor::extractWordsFromMessage(message, words);

        LOG_DEBUG(<< "Message: " << message);
        LOG_DEBUG(<< "Words: " << words);

        BOOST_CHECK_EQUAL(std::string("Found corresponding"), words);
    }
    {
        std::string message("]\", which is more than the configured time (StuckThreadMaxTime) of \"600\" seconds. Stack trace:");
        std::string words;

        ml::core::CWordExtractor::extractWordsFromMessage(message, words);

        LOG_DEBUG(<< "Message: " << message);
        LOG_DEBUG(<< "Words: " << words);

        BOOST_CHECK_EQUAL(std::string("which is more than the configured time of seconds Stack trace"),
                          words);
    }
}

BOOST_AUTO_TEST_CASE(testMinConsecutive) {
    {
        std::string message("2017-01-25 02:10:03,551 ERROR [co.elastic.tradefeedtracker.MessageLoggerService] Failed to Rollback");
        std::string words;

        ml::core::CWordExtractor::extractWordsFromMessage(2, message, words);

        LOG_DEBUG(<< "Min consecutive: 2");
        LOG_DEBUG(<< "Message: " << message);
        LOG_DEBUG(<< "Words: " << words);

        BOOST_CHECK_EQUAL(std::string("Failed to Rollback"), words);

        ml::core::CWordExtractor::extractWordsFromMessage(3, message, words);

        LOG_DEBUG(<< "Min consecutive: 3");
        LOG_DEBUG(<< "Message: " << message);
        LOG_DEBUG(<< "Words: " << words);

        BOOST_CHECK_EQUAL(std::string("Failed to Rollback"), words);

        ml::core::CWordExtractor::extractWordsFromMessage(4, message, words);

        LOG_DEBUG(<< "Min consecutive: 4");
        LOG_DEBUG(<< "Message: " << message);
        LOG_DEBUG(<< "Words: " << words);

        BOOST_CHECK_EQUAL(std::string(""), words);
    }
    {
        std::string message("]\", which is more than the configured time (StuckThreadMaxTime) of \"600\" seconds. Stack trace:");
        std::string words;

        ml::core::CWordExtractor::extractWordsFromMessage(2, message, words);

        LOG_DEBUG(<< "Min consecutive: 2");
        LOG_DEBUG(<< "Message: " << message);
        LOG_DEBUG(<< "Words: " << words);

        BOOST_CHECK_EQUAL(std::string("which is more than the configured time seconds Stack trace"),
                          words);

        ml::core::CWordExtractor::extractWordsFromMessage(3, message, words);

        LOG_DEBUG(<< "Min consecutive: 3");
        LOG_DEBUG(<< "Message: " << message);
        LOG_DEBUG(<< "Words: " << words);

        BOOST_CHECK_EQUAL(std::string("which is more than the configured time seconds Stack trace"),
                          words);

        ml::core::CWordExtractor::extractWordsFromMessage(4, message, words);

        LOG_DEBUG(<< "Min consecutive: 4");
        LOG_DEBUG(<< "Message: " << message);
        LOG_DEBUG(<< "Words: " << words);

        BOOST_CHECK_EQUAL(std::string("which is more than the configured time"), words);
    }
    {
        std::string message("<ml00-4253.1.p2ps: Warning: > Output threshold breached for: dave at position 192.168.156.136/net using "
                            "application 163 on channel 12.<END>");
        std::string words;

        ml::core::CWordExtractor::extractWordsFromMessage(2, message, words);

        LOG_DEBUG(<< "Min consecutive: 2");
        LOG_DEBUG(<< "Message: " << message);
        LOG_DEBUG(<< "Words: " << words);

        BOOST_CHECK_EQUAL(std::string("Output threshold breached for at position using application on channel"),
                          words);

        ml::core::CWordExtractor::extractWordsFromMessage(3, message, words);

        LOG_DEBUG(<< "Min consecutive: 3");
        LOG_DEBUG(<< "Message: " << message);
        LOG_DEBUG(<< "Words: " << words);

        BOOST_CHECK_EQUAL(std::string("Output threshold breached for"), words);

        ml::core::CWordExtractor::extractWordsFromMessage(4, message, words);

        LOG_DEBUG(<< "Min consecutive: 4");
        LOG_DEBUG(<< "Message: " << message);
        LOG_DEBUG(<< "Words: " << words);

        BOOST_CHECK_EQUAL(std::string("Output threshold breached for"), words);
    }
}

BOOST_AUTO_TEST_SUITE_END()
