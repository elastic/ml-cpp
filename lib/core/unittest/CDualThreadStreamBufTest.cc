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
#include "CDualThreadStreamBufTest.h"

#include <core/CDualThreadStreamBuf.h>
#include <core/CLogger.h>
#include <core/CSleep.h>
#include <core/CThread.h>
#include <core/CTimeUtils.h>
#include <core/CoreTypes.h>

#include <algorithm>
#include <istream>
#include <string>

#include <stdint.h>
#include <string.h>

CppUnit::Test* CDualThreadStreamBufTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CDualThreadStreamBufTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDualThreadStreamBufTest>("CDualThreadStreamBufTest::testThroughput",
                                                                            &CDualThreadStreamBufTest::testThroughput));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDualThreadStreamBufTest>("CDualThreadStreamBufTest::testSlowConsumer",
                                                                            &CDualThreadStreamBufTest::testSlowConsumer));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CDualThreadStreamBufTest>("CDualThreadStreamBufTest::testPutback", &CDualThreadStreamBufTest::testPutback));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CDualThreadStreamBufTest>("CDualThreadStreamBufTest::testFatal", &CDualThreadStreamBufTest::testFatal));

    return suiteOfTests;
}

namespace {

class CInputThread : public ml::core::CThread {
public:
    CInputThread(ml::core::CDualThreadStreamBuf& buffer, uint32_t delay = 0, size_t fatalAfter = 0)
        : m_Buffer(buffer), m_Delay(delay), m_FatalAfter(fatalAfter), m_TotalData(0) {}

    size_t totalData() const { return m_TotalData; }

protected:
    virtual void run() {
        std::istream strm(&m_Buffer);
        size_t count(0);
        std::string line;
        while (std::getline(strm, line)) {
            ++count;
            m_TotalData += line.length();
            ++m_TotalData; // For the delimiter
            CPPUNIT_ASSERT_EQUAL(static_cast<std::streampos>(m_TotalData), strm.tellg());
            ml::core::CSleep::sleep(m_Delay);
            if (count == m_FatalAfter) {
                m_Buffer.signalFatalError();
            }
        }
    }

    virtual void shutdown() { m_Buffer.signalFatalError(); }

private:
    ml::core::CDualThreadStreamBuf& m_Buffer;
    uint32_t m_Delay;
    size_t m_FatalAfter;
    size_t m_TotalData;
};

const char* DATA("According to the most recent Wikipedia definition \"Predictive "
                 "analytics encompasses a variety of statistical techniques from "
                 "modeling, machine learning, data mining and game theory that ... "
                 "exploit patterns found in historical and transactional data to "
                 "identify risks and opportunities.\"\n"
                 "In applications such as credit scoring, predictive analytics "
                 "identifies patterns and relationships in huge volumes of data, hidden "
                 "to human analysis, that presages an undesirable outcome.  Many "
                 "vendors refer to their ability to project a ramp in a single metric, "
                 "say CPU utilization, as predictive analytics.  As most users know, "
                 "these capabilities are of limited value in that single metrics are "
                 "rarely the cause of cataclysmic failures.  Rather it is the impact of "
                 "change between components that causes failure in complex IT systems.\n");
}

void CDualThreadStreamBufTest::testThroughput() {
    static const size_t TEST_SIZE(1000000);
    size_t dataSize(::strlen(DATA));
    size_t totalDataSize(TEST_SIZE * dataSize);

    ml::core::CDualThreadStreamBuf buf;
    CInputThread inputThread(buf);
    inputThread.start();

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting REST buffer throughput test at " << ml::core::CTimeUtils::toTimeString(start));

    for (size_t count = 0; count < TEST_SIZE; ++count) {
        std::streamsize toWrite(static_cast<std::streamsize>(dataSize));
        const char* ptr(DATA);
        while (toWrite > 0) {
            std::streamsize written(buf.sputn(ptr, toWrite));
            CPPUNIT_ASSERT(written > 0);
            toWrite -= written;
            ptr += written;
        }
    }

    CPPUNIT_ASSERT_EQUAL(static_cast<std::streampos>(totalDataSize), buf.pubseekoff(0, std::ios_base::cur, std::ios_base::out));

    buf.signalEndOfFile();

    inputThread.waitForFinish();

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO("Finished REST buffer throughput test at " << ml::core::CTimeUtils::toTimeString(end));

    CPPUNIT_ASSERT_EQUAL(totalDataSize, inputThread.totalData());

    LOG_INFO("REST buffer throughput test with test size " << TEST_SIZE << " (total data transferred " << totalDataSize << " bytes) took "
                                                           << (end - start) << " seconds");
}

void CDualThreadStreamBufTest::testSlowConsumer() {
    static const size_t TEST_SIZE(25);
    static const uint32_t DELAY(200);
    size_t dataSize(::strlen(DATA));
    size_t numNewLines(std::count(DATA, DATA + dataSize, '\n'));
    size_t totalDataSize(TEST_SIZE * dataSize);

    ml::core::CDualThreadStreamBuf buf;
    CInputThread inputThread(buf, DELAY);
    inputThread.start();

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting REST buffer slow consumer test at " << ml::core::CTimeUtils::toTimeString(start));

    for (size_t count = 0; count < TEST_SIZE; ++count) {
        std::streamsize toWrite(static_cast<std::streamsize>(dataSize));
        const char* ptr(DATA);
        while (toWrite > 0) {
            std::streamsize written(buf.sputn(ptr, toWrite));
            CPPUNIT_ASSERT(written > 0);
            toWrite -= written;
            ptr += written;
        }
    }

    buf.signalEndOfFile();

    inputThread.waitForFinish();

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO("Finished REST buffer slow consumer test at " << ml::core::CTimeUtils::toTimeString(end));

    CPPUNIT_ASSERT_EQUAL(totalDataSize, inputThread.totalData());

    ml::core_t::TTime duration(end - start);
    LOG_INFO("REST buffer slow consumer test with test size " << TEST_SIZE << ", " << numNewLines << " newlines per message and delay "
                                                              << DELAY << "ms took " << duration << " seconds");

    ml::core_t::TTime delaySecs(static_cast<ml::core_t::TTime>((DELAY * numNewLines * TEST_SIZE) / 1000));
    CPPUNIT_ASSERT(duration >= delaySecs);
    static const ml::core_t::TTime TOLERANCE(3);
    CPPUNIT_ASSERT(duration <= delaySecs + TOLERANCE);
}

void CDualThreadStreamBufTest::testPutback() {
    size_t dataSize(::strlen(DATA));

    ml::core::CDualThreadStreamBuf buf;

    std::streamsize toWrite(static_cast<std::streamsize>(dataSize));
    const char* ptr(DATA);
    while (toWrite > 0) {
        std::streamsize written(buf.sputn(ptr, toWrite));
        CPPUNIT_ASSERT(written > 0);
        toWrite -= written;
        ptr += written;
    }

    buf.signalEndOfFile();

    static const char* PUTBACK_CHARS("put this back");
    std::istream strm(&buf);
    char c('\0');
    CPPUNIT_ASSERT(strm.get(c).good());
    CPPUNIT_ASSERT_EQUAL(*DATA, c);
    CPPUNIT_ASSERT(strm.putback(c).good());
    for (const char* putbackChar = PUTBACK_CHARS; *putbackChar != '\0'; ++putbackChar) {
        CPPUNIT_ASSERT(strm.putback(*putbackChar).good());
    }
    std::string actual;
    for (const char* putbackChar = PUTBACK_CHARS; *putbackChar != '\0'; ++putbackChar) {
        CPPUNIT_ASSERT(strm.get(c).good());
        actual.insert(actual.begin(), c);
    }
    CPPUNIT_ASSERT_EQUAL(std::string(PUTBACK_CHARS), actual);

    std::string remainder;
    std::string line;
    while (std::getline(strm, line)) {
        remainder += line;
        remainder += '\n';
    }
    CPPUNIT_ASSERT_EQUAL(std::string(DATA), remainder);
}

void CDualThreadStreamBufTest::testFatal() {
    static const size_t TEST_SIZE(10000);
    static const size_t BUFFER_CAPACITY(16384);
    size_t dataSize(::strlen(DATA));

    // These conditions need to be true for the test to work properly
    CPPUNIT_ASSERT(dataSize < BUFFER_CAPACITY);
    CPPUNIT_ASSERT(BUFFER_CAPACITY * 3 < TEST_SIZE * dataSize);

    ml::core::CDualThreadStreamBuf buf(BUFFER_CAPACITY);
    CInputThread inputThread(buf, 1000, 1);
    inputThread.start();

    size_t totalDataWritten(0);
    for (size_t count = 0; count < TEST_SIZE; ++count) {
        std::streamsize toWrite(static_cast<std::streamsize>(dataSize));
        const char* ptr(DATA);
        while (toWrite > 0) {
            std::streamsize written(buf.sputn(ptr, toWrite));
            if (written == 0) {
                break;
            }
            toWrite -= written;
            ptr += written;
            totalDataWritten += static_cast<size_t>(written);
        }
    }

    buf.signalEndOfFile();

    inputThread.waitForFinish();

    LOG_DEBUG("Total data written in fatal error test of size " << TEST_SIZE << " is " << totalDataWritten << " bytes");

    // The fatal error should have stopped the writer thread from writing all the data
    CPPUNIT_ASSERT(totalDataWritten >= BUFFER_CAPACITY);
    CPPUNIT_ASSERT(totalDataWritten <= 3 * BUFFER_CAPACITY);
}
