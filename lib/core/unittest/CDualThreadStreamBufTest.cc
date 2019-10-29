/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDualThreadStreamBuf.h>
#include <core/CLogger.h>
#include <core/CSleep.h>
#include <core/CThread.h>
#include <core/CTimeUtils.h>
#include <core/CoreTypes.h>

#include <boost/optional.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <istream>
#include <string>

#include <stdint.h>
#include <string.h>

BOOST_AUTO_TEST_SUITE(CDualThreadStreamBufTest)

namespace {

class CInputThread : public ml::core::CThread {
private:
    class CTellgStore {
    public:
        CTellgStore(std::streampos tellg) : m_Tellg{tellg} {}
        std::streampos tellg() { return m_Tellg; }

    private:
        std::streampos m_Tellg;
    };

    using TOptionalTellgStore = boost::optional<CTellgStore>;

public:
    CInputThread(ml::core::CDualThreadStreamBuf& buffer, uint32_t delay = 0, size_t fatalAfter = 0)
        : m_Buffer(buffer), m_Delay(delay), m_FatalAfter(fatalAfter),
          m_TotalData(0) {}

    size_t totalData() const { return m_TotalData; }

    void propagateLastDetectedMismatch() {
        if (m_LastMismatch.has_value()) {
            // Reconstruct the original local variable to make any
            // assertion failure message easier to understand
            CTellgStore strm{*m_LastMismatch};
            BOOST_REQUIRE_EQUAL(static_cast<std::streampos>(m_TotalData), strm.tellg());
        }
    }

protected:
    virtual void run() {
        std::istream strm(&m_Buffer);
        size_t count(0);
        std::string line;
        while (std::getline(strm, line)) {
            ++count;
            m_TotalData += line.length();
            ++m_TotalData; // For the delimiter
            if (static_cast<std::streampos>(m_TotalData) != strm.tellg()) {
                m_LastMismatch = CTellgStore(strm.tellg());
                m_Buffer.signalFatalError();
                break;
            }
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
    TOptionalTellgStore m_LastMismatch;
};

const char*
    DATA("According to the most recent Wikipedia definition \"Predictive "
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

BOOST_AUTO_TEST_CASE(testThroughput) {
    static const size_t TEST_SIZE(1000000);
    size_t dataSize(::strlen(DATA));
    size_t totalDataSize(TEST_SIZE * dataSize);

    ml::core::CDualThreadStreamBuf buf;
    CInputThread inputThread(buf);
    inputThread.start();

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting REST buffer throughput test at "
             << ml::core::CTimeUtils::toTimeString(start));

    for (size_t count = 0; count < TEST_SIZE; ++count) {
        std::streamsize toWrite(static_cast<std::streamsize>(dataSize));
        const char* ptr(DATA);
        while (toWrite > 0) {
            std::streamsize written(buf.sputn(ptr, toWrite));
            BOOST_TEST_REQUIRE(written > 0);
            toWrite -= written;
            ptr += written;
        }
    }

    BOOST_REQUIRE_EQUAL(static_cast<std::streampos>(totalDataSize),
                        buf.pubseekoff(0, std::ios_base::cur, std::ios_base::out));

    buf.signalEndOfFile();

    inputThread.waitForFinish();
    inputThread.propagateLastDetectedMismatch();

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished REST buffer throughput test at "
             << ml::core::CTimeUtils::toTimeString(end));

    BOOST_REQUIRE_EQUAL(totalDataSize, inputThread.totalData());

    LOG_INFO(<< "REST buffer throughput test with test size " << TEST_SIZE << " (total data transferred "
             << totalDataSize << " bytes) took " << (end - start) << " seconds");
}

BOOST_AUTO_TEST_CASE(testSlowConsumer) {
    static const size_t TEST_SIZE(25);
    static const uint32_t DELAY(200);
    size_t dataSize(::strlen(DATA));
    size_t numNewLines(std::count(DATA, DATA + dataSize, '\n'));
    size_t totalDataSize(TEST_SIZE * dataSize);

    ml::core::CDualThreadStreamBuf buf;
    CInputThread inputThread(buf, DELAY);
    inputThread.start();

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting REST buffer slow consumer test at "
             << ml::core::CTimeUtils::toTimeString(start));

    for (size_t count = 0; count < TEST_SIZE; ++count) {
        std::streamsize toWrite(static_cast<std::streamsize>(dataSize));
        const char* ptr(DATA);
        while (toWrite > 0) {
            std::streamsize written(buf.sputn(ptr, toWrite));
            BOOST_TEST_REQUIRE(written > 0);
            toWrite -= written;
            ptr += written;
        }
    }

    buf.signalEndOfFile();

    inputThread.waitForFinish();
    inputThread.propagateLastDetectedMismatch();

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished REST buffer slow consumer test at "
             << ml::core::CTimeUtils::toTimeString(end));

    BOOST_REQUIRE_EQUAL(totalDataSize, inputThread.totalData());

    ml::core_t::TTime duration(end - start);
    LOG_INFO(<< "REST buffer slow consumer test with test size " << TEST_SIZE
             << ", " << numNewLines << " newlines per message and delay "
             << DELAY << "ms took " << duration << " seconds");

    ml::core_t::TTime delaySecs(
        static_cast<ml::core_t::TTime>((DELAY * numNewLines * TEST_SIZE) / 1000));
    BOOST_TEST_REQUIRE(duration >= delaySecs);
    static const ml::core_t::TTime TOLERANCE(3);
    BOOST_TEST_REQUIRE(duration <= delaySecs + TOLERANCE);
}

BOOST_AUTO_TEST_CASE(testPutback) {
    size_t dataSize(::strlen(DATA));

    ml::core::CDualThreadStreamBuf buf;

    std::streamsize toWrite(static_cast<std::streamsize>(dataSize));
    const char* ptr(DATA);
    while (toWrite > 0) {
        std::streamsize written(buf.sputn(ptr, toWrite));
        BOOST_TEST_REQUIRE(written > 0);
        toWrite -= written;
        ptr += written;
    }

    buf.signalEndOfFile();

    static const char* const PUTBACK_CHARS("put this back");
    std::istream strm(&buf);
    char c('\0');
    BOOST_TEST_REQUIRE(strm.get(c).good());
    BOOST_REQUIRE_EQUAL(*DATA, c);
    BOOST_TEST_REQUIRE(strm.putback(c).good());
    for (const char* putbackChar = PUTBACK_CHARS; *putbackChar != '\0'; ++putbackChar) {
        BOOST_TEST_REQUIRE(strm.putback(*putbackChar).good());
    }
    std::string actual;
    for (const char* putbackChar = PUTBACK_CHARS; *putbackChar != '\0'; ++putbackChar) {
        BOOST_TEST_REQUIRE(strm.get(c).good());
        actual.insert(actual.begin(), c);
    }
    BOOST_REQUIRE_EQUAL(std::string(PUTBACK_CHARS), actual);

    std::string remainder;
    std::string line;
    while (std::getline(strm, line)) {
        remainder += line;
        remainder += '\n';
    }
    BOOST_REQUIRE_EQUAL(std::string(DATA), remainder);
}

BOOST_AUTO_TEST_CASE(testFatal) {
    static const size_t TEST_SIZE(10000);
    static const size_t BUFFER_CAPACITY(16384);
    size_t dataSize(::strlen(DATA));

    // These conditions need to be true for the test to work properly
    BOOST_TEST_REQUIRE(dataSize < BUFFER_CAPACITY);
    BOOST_TEST_REQUIRE(BUFFER_CAPACITY * 3 < TEST_SIZE * dataSize);

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
    inputThread.propagateLastDetectedMismatch();

    LOG_DEBUG(<< "Total data written in fatal error test of size " << TEST_SIZE
              << " is " << totalDataWritten << " bytes");

    // The fatal error should have stopped the writer thread from writing all the data
    BOOST_TEST_REQUIRE(totalDataWritten >= BUFFER_CAPACITY);
    BOOST_TEST_REQUIRE(totalDataWritten <= 3 * BUFFER_CAPACITY);
}

BOOST_AUTO_TEST_SUITE_END()
