/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include "CWordExtractorTest.h"

#include <core/CLogger.h>
#include <core/CWordExtractor.h>

CppUnit::Test *CWordExtractorTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CWordExtractorTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CWordExtractorTest>(
        "CWordExtractorTest::testWordExtract", &CWordExtractorTest::testWordExtract));
    suiteOfTests->addTest(new CppUnit::TestCaller<CWordExtractorTest>(
        "CWordExtractorTest::testMinConsecutive", &CWordExtractorTest::testMinConsecutive));

    return suiteOfTests;
}

void CWordExtractorTest::testWordExtract(void) {
    {
        std::string message("2017-01-25 02:10:03,551 ERROR "
                            "[co.elastic.tradefeedtracker.MessageLoggerService] Failed to "
                            "Rollback");
        std::string words;

        ml::core::CWordExtractor::extractWordsFromMessage(message, words);

        LOG_DEBUG("Message: " << message);
        LOG_DEBUG("Words: " << words);

        CPPUNIT_ASSERT_EQUAL(std::string("Failed to Rollback"), words);
    }
    {
        std::string message("2017-01-25 14:20:49,646 INFO  "
                            "[co.elastic.settlement.synchronization.errors."
                            "NonFXInstructionSyncImpl] Found corresponding outgoingPaymentFlow :: "
                            "OutGoingPaymentFlow.id = 7480");
        std::string words;

        ml::core::CWordExtractor::extractWordsFromMessage(message, words);

        LOG_DEBUG("Message: " << message);
        LOG_DEBUG("Words: " << words);

        CPPUNIT_ASSERT_EQUAL(std::string("Found corresponding"), words);
    }
    {
        std::string message("]\", which is more than the configured time (StuckThreadMaxTime) of "
                            "\"600\" seconds. Stack trace:");
        std::string words;

        ml::core::CWordExtractor::extractWordsFromMessage(message, words);

        LOG_DEBUG("Message: " << message);
        LOG_DEBUG("Words: " << words);

        CPPUNIT_ASSERT_EQUAL(
            std::string("which is more than the configured time of seconds Stack trace"), words);
    }
}

void CWordExtractorTest::testMinConsecutive(void) {
    {
        std::string message("2017-01-25 02:10:03,551 ERROR "
                            "[co.elastic.tradefeedtracker.MessageLoggerService] Failed to "
                            "Rollback");
        std::string words;

        ml::core::CWordExtractor::extractWordsFromMessage(2, message, words);

        LOG_DEBUG("Min consecutive: 2");
        LOG_DEBUG("Message: " << message);
        LOG_DEBUG("Words: " << words);

        CPPUNIT_ASSERT_EQUAL(std::string("Failed to Rollback"), words);

        ml::core::CWordExtractor::extractWordsFromMessage(3, message, words);

        LOG_DEBUG("Min consecutive: 3");
        LOG_DEBUG("Message: " << message);
        LOG_DEBUG("Words: " << words);

        CPPUNIT_ASSERT_EQUAL(std::string("Failed to Rollback"), words);

        ml::core::CWordExtractor::extractWordsFromMessage(4, message, words);

        LOG_DEBUG("Min consecutive: 4");
        LOG_DEBUG("Message: " << message);
        LOG_DEBUG("Words: " << words);

        CPPUNIT_ASSERT_EQUAL(std::string(""), words);
    }
    {
        std::string message("]\", which is more than the configured time (StuckThreadMaxTime) of "
                            "\"600\" seconds. Stack trace:");
        std::string words;

        ml::core::CWordExtractor::extractWordsFromMessage(2, message, words);

        LOG_DEBUG("Min consecutive: 2");
        LOG_DEBUG("Message: " << message);
        LOG_DEBUG("Words: " << words);

        CPPUNIT_ASSERT_EQUAL(
            std::string("which is more than the configured time seconds Stack trace"), words);

        ml::core::CWordExtractor::extractWordsFromMessage(3, message, words);

        LOG_DEBUG("Min consecutive: 3");
        LOG_DEBUG("Message: " << message);
        LOG_DEBUG("Words: " << words);

        CPPUNIT_ASSERT_EQUAL(
            std::string("which is more than the configured time seconds Stack trace"), words);

        ml::core::CWordExtractor::extractWordsFromMessage(4, message, words);

        LOG_DEBUG("Min consecutive: 4");
        LOG_DEBUG("Message: " << message);
        LOG_DEBUG("Words: " << words);

        CPPUNIT_ASSERT_EQUAL(std::string("which is more than the configured time"), words);
    }
    {
        std::string message("<ml00-4253.1.p2ps: Warning: > Output threshold breached for: dave at "
                            "position 192.168.156.136/net using application 163 on channel "
                            "12.<END>");
        std::string words;

        ml::core::CWordExtractor::extractWordsFromMessage(2, message, words);

        LOG_DEBUG("Min consecutive: 2");
        LOG_DEBUG("Message: " << message);
        LOG_DEBUG("Words: " << words);

        CPPUNIT_ASSERT_EQUAL(
            std::string("Output threshold breached for at position using application on channel"),
            words);

        ml::core::CWordExtractor::extractWordsFromMessage(3, message, words);

        LOG_DEBUG("Min consecutive: 3");
        LOG_DEBUG("Message: " << message);
        LOG_DEBUG("Words: " << words);

        CPPUNIT_ASSERT_EQUAL(std::string("Output threshold breached for"), words);

        ml::core::CWordExtractor::extractWordsFromMessage(4, message, words);

        LOG_DEBUG("Min consecutive: 4");
        LOG_DEBUG("Message: " << message);
        LOG_DEBUG("Words: " << words);

        CPPUNIT_ASSERT_EQUAL(std::string("Output threshold breached for"), words);
    }
}
