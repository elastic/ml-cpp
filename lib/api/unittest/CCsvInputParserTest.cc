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
#include "CCsvInputParserTest.h"

#include <core/CLogger.h>
#include <core/CoreTypes.h>
#include <core/CStringUtils.h>
#include <core/CTimeUtils.h>
#include <core/CTimezone.h>

#include <api/CCsvInputParser.h>

#include <boost/range.hpp>

#include <algorithm>
#include <fstream>
#include <functional>
#include <vector>


CppUnit::Test *CCsvInputParserTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CCsvInputParserTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CCsvInputParserTest>(
                               "CCsvInputParserTest::testSimpleDelims",
                               &CCsvInputParserTest::testSimpleDelims) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CCsvInputParserTest>(
                               "CCsvInputParserTest::testComplexDelims",
                               &CCsvInputParserTest::testComplexDelims) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CCsvInputParserTest>(
                               "CCsvInputParserTest::testThroughput",
                               &CCsvInputParserTest::testThroughput) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CCsvInputParserTest>(
                               "CCsvInputParserTest::testDateParse",
                               &CCsvInputParserTest::testDateParse) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CCsvInputParserTest>(
                               "CCsvInputParserTest::testQuoteParsing",
                               &CCsvInputParserTest::testQuoteParsing) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CCsvInputParserTest>(
                               "CCsvInputParserTest::testLineParser",
                               &CCsvInputParserTest::testLineParser) );

    return suiteOfTests;
}

namespace {


class CVisitor {
    public:
        CVisitor(void)
            : m_Fast(true),
              m_RecordCount(0)
        {}

        CVisitor(const ml::api::CCsvInputParser::TStrVec &expectedFieldNames)
            : m_Fast(false),
              m_RecordCount(0),
              m_ExpectedFieldNames(expectedFieldNames)
        {}

        //! Handle a record
        bool operator()(const ml::api::CCsvInputParser::TStrStrUMap &dataRowFields) {
            ++m_RecordCount;

            // For the throughput test, the assertions below will skew the
            // results, so bypass them
            if (m_Fast) {
                return true;
            }

            // Check the field names
            for (const auto &entry : dataRowFields) {
                auto iter = std::find(m_ExpectedFieldNames.begin(), m_ExpectedFieldNames.end(), entry.first);
                CPPUNIT_ASSERT(iter != m_ExpectedFieldNames.end());
            }

            CPPUNIT_ASSERT_EQUAL(m_ExpectedFieldNames.size(), dataRowFields.size());

            // Check the line count is consistent with the _raw field
            ml::api::CCsvInputParser::TStrStrUMapCItr rawIter = dataRowFields.find("_raw");
            CPPUNIT_ASSERT(rawIter != dataRowFields.end());
            ml::api::CCsvInputParser::TStrStrUMapCItr lineCountIter = dataRowFields.find("linecount");
            CPPUNIT_ASSERT(lineCountIter != dataRowFields.end());

            size_t expectedLineCount(1 + std::count(rawIter->second.begin(),
                                                    rawIter->second.end(),
                                                    '\n'));
            size_t lineCount(0);
            CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType(lineCountIter->second, lineCount));
            CPPUNIT_ASSERT_EQUAL(expectedLineCount, lineCount);

            return true;
        }

        size_t recordCount(void) const {
            return m_RecordCount;
        }

    private:
        bool m_Fast;
        size_t m_RecordCount;
        ml::api::CCsvInputParser::TStrVec m_ExpectedFieldNames;
};

class CTimeCheckingVisitor {
    public:
        typedef std::vector<ml::core_t::TTime> TTimeVec;

    public:
        CTimeCheckingVisitor(const std::string &timeField,
                             const std::string &timeFormat,
                             const TTimeVec &expectedTimes)
            : m_RecordCount(0),
              m_TimeField(timeField),
              m_TimeFormat(timeFormat),
              m_ExpectedTimes(expectedTimes)
        {}

        //! Handle a record
        bool operator()(const ml::api::CCsvInputParser::TStrStrUMap &dataRowFields) {
            // Check the time field exists
            CPPUNIT_ASSERT(m_RecordCount < m_ExpectedTimes.size());

            auto iter = dataRowFields.find(m_TimeField);
            CPPUNIT_ASSERT(iter != dataRowFields.end());

            // Now check the actual time
            ml::api::CCsvInputParser::TStrStrUMapCItr fieldIter = dataRowFields.find(m_TimeField);
            CPPUNIT_ASSERT(fieldIter != dataRowFields.end());
            ml::core_t::TTime timeVal(0);
            if (m_TimeFormat.empty()) {
                CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType(fieldIter->second,
                                                                    timeVal));
            } else   {
                CPPUNIT_ASSERT(ml::core::CTimeUtils::strptime(m_TimeFormat,
                                                              fieldIter->second,
                                                              timeVal));
                LOG_DEBUG("Converted " << fieldIter->second <<
                          " to " << timeVal <<
                          " using format " << m_TimeFormat);
            }
            CPPUNIT_ASSERT_EQUAL(m_ExpectedTimes[m_RecordCount], timeVal);

            ++m_RecordCount;

            return true;
        }

        size_t recordCount(void) const {
            return m_RecordCount;
        }

    private:
        size_t m_RecordCount;
        std::string m_TimeField;
        std::string m_TimeFormat;
        TTimeVec m_ExpectedTimes;
};

class CQuoteCheckingVisitor {
    public:
        CQuoteCheckingVisitor(void)
            : m_RecordCount(0)
        {}

        //! Handle a record
        bool operator()(const ml::api::CCsvInputParser::TStrStrUMap &dataRowFields) {
            // Now check quoted fields
            ml::api::CCsvInputParser::TStrStrUMapCItr fieldIter = dataRowFields.find("q1");
            CPPUNIT_ASSERT(fieldIter != dataRowFields.end());
            CPPUNIT_ASSERT_EQUAL(std::string(""), fieldIter->second);

            fieldIter = dataRowFields.find("q2");
            CPPUNIT_ASSERT(fieldIter != dataRowFields.end());
            CPPUNIT_ASSERT_EQUAL(std::string(""), fieldIter->second);

            fieldIter = dataRowFields.find("q3");
            CPPUNIT_ASSERT(fieldIter != dataRowFields.end());
            CPPUNIT_ASSERT_EQUAL(std::string("\""), fieldIter->second);

            fieldIter = dataRowFields.find("q4");
            CPPUNIT_ASSERT(fieldIter != dataRowFields.end());
            CPPUNIT_ASSERT_EQUAL(std::string("\"\""), fieldIter->second);

            ++m_RecordCount;

            return true;
        }

        size_t recordCount(void) const {
            return m_RecordCount;
        }

    private:
        size_t m_RecordCount;
};


}

void CCsvInputParserTest::testSimpleDelims(void) {
    std::ifstream simpleStrm("testfiles/simple.txt");
    CPPUNIT_ASSERT(simpleStrm.is_open());

    ml::api::CCsvInputParser parser(simpleStrm);

    ml::api::CCsvInputParser::TStrVec expectedFieldNames;
    expectedFieldNames.push_back("_cd");
    expectedFieldNames.push_back("_indextime");
    expectedFieldNames.push_back("_kv");
    expectedFieldNames.push_back("_raw");
    expectedFieldNames.push_back("_serial");
    expectedFieldNames.push_back("_si");
    expectedFieldNames.push_back("_sourcetype");
    expectedFieldNames.push_back("_time");
    expectedFieldNames.push_back("date_hour");
    expectedFieldNames.push_back("date_mday");
    expectedFieldNames.push_back("date_minute");
    expectedFieldNames.push_back("date_month");
    expectedFieldNames.push_back("date_second");
    expectedFieldNames.push_back("date_wday");
    expectedFieldNames.push_back("date_year");
    expectedFieldNames.push_back("date_zone");
    expectedFieldNames.push_back("eventtype");
    expectedFieldNames.push_back("host");
    expectedFieldNames.push_back("index");
    expectedFieldNames.push_back("linecount");
    expectedFieldNames.push_back("punct");
    expectedFieldNames.push_back("source");
    expectedFieldNames.push_back("sourcetype");
    expectedFieldNames.push_back("server");
    expectedFieldNames.push_back("timeendpos");
    expectedFieldNames.push_back("timestartpos");

    CVisitor visitor(expectedFieldNames);

    CPPUNIT_ASSERT(parser.readStream(std::ref(visitor)));

    CPPUNIT_ASSERT_EQUAL(size_t(15), visitor.recordCount());
}

void CCsvInputParserTest::testComplexDelims(void) {
    std::ifstream complexStrm("testfiles/complex.txt");
    CPPUNIT_ASSERT(complexStrm.is_open());

    ml::api::CCsvInputParser parser(complexStrm);

    ml::api::CCsvInputParser::TStrVec expectedFieldNames;
    expectedFieldNames.push_back("_cd");
    expectedFieldNames.push_back("_indextime");
    expectedFieldNames.push_back("_kv");
    expectedFieldNames.push_back("_raw");
    expectedFieldNames.push_back("_serial");
    expectedFieldNames.push_back("_si");
    expectedFieldNames.push_back("_sourcetype");
    expectedFieldNames.push_back("_time");
    expectedFieldNames.push_back("date_hour");
    expectedFieldNames.push_back("date_mday");
    expectedFieldNames.push_back("date_minute");
    expectedFieldNames.push_back("date_month");
    expectedFieldNames.push_back("date_second");
    expectedFieldNames.push_back("date_wday");
    expectedFieldNames.push_back("date_year");
    expectedFieldNames.push_back("date_zone");
    expectedFieldNames.push_back("eventtype");
    expectedFieldNames.push_back("host");
    expectedFieldNames.push_back("index");
    expectedFieldNames.push_back("linecount");
    expectedFieldNames.push_back("punct");
    expectedFieldNames.push_back("source");
    expectedFieldNames.push_back("sourcetype");
    expectedFieldNames.push_back("server");
    expectedFieldNames.push_back("timeendpos");
    expectedFieldNames.push_back("timestartpos");

    CVisitor visitor(expectedFieldNames);

    CPPUNIT_ASSERT(parser.readStream(std::ref(visitor)));
}

void CCsvInputParserTest::testThroughput(void) {
    std::ifstream ifs("testfiles/simple.txt");
    CPPUNIT_ASSERT(ifs.is_open());

    std::string line;

    std::string header;
    if (std::getline(ifs, line).good()) {
        header = line;
        header += '\n';
    }

    std::string restOfFile;
    size_t nonHeaderLines(0);
    while (std::getline(ifs, line).good()) {
        if (line.empty()) {
            break;
        }
        ++nonHeaderLines;
        restOfFile += line;
        restOfFile += '\n';
    }

    // Assume there are two lines per record in the input file
    CPPUNIT_ASSERT((nonHeaderLines % 2) == 0);
    size_t recordsPerBlock(nonHeaderLines / 2);

    // Construct a large test input
    static const size_t TEST_SIZE(10000);
    std::string input(header);
    for (size_t count = 0; count < TEST_SIZE; ++count) {
        input += restOfFile;
    }
    LOG_DEBUG("Input size is " << input.length());

    ml::api::CCsvInputParser parser(input);

    CVisitor visitor;

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting throughput test at " <<
             ml::core::CTimeUtils::toTimeString(start));

    CPPUNIT_ASSERT(parser.readStream(std::ref(visitor)));

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO("Finished throughput test at " <<
             ml::core::CTimeUtils::toTimeString(end));

    CPPUNIT_ASSERT_EQUAL(recordsPerBlock * TEST_SIZE, visitor.recordCount());

    LOG_INFO("Parsing " << visitor.recordCount() <<
             " records took " << (end - start) << " seconds");
}

void CCsvInputParserTest::testDateParse(void) {
    static const ml::core_t::TTime EXPECTED_TIMES[] = {
        1359331200,
        1359331200,
        1359331207,
        1359331220,
        1359331259,
        1359331262,
        1359331269,
        1359331270,
        1359331272,
        1359331296,
        1359331301,
        1359331311,
        1359331314,
        1359331315,
        1359331316,
        1359331321,
        1359331328,
        1359331333,
        1359331349,
        1359331352,
        1359331370,
        1359331382,
        1359331385,
        1359331386,
        1359331395,
        1359331404,
        1359331416,
        1359331416,
        1359331424,
        1359331429
    };

    CTimeCheckingVisitor::TTimeVec expectedTimes(boost::begin(EXPECTED_TIMES),
                                                 boost::end(EXPECTED_TIMES));

    // Ensure we are in UK timewise
    CPPUNIT_ASSERT(ml::core::CTimezone::setTimezone("Europe/London"));

    {
        std::ifstream csvStrm("testfiles/s.csv");
        CPPUNIT_ASSERT(csvStrm.is_open());

        CTimeCheckingVisitor visitor("time",
                                     "",
                                     expectedTimes);

        ml::api::CCsvInputParser parser(csvStrm);

        CPPUNIT_ASSERT(parser.readStream(std::ref(visitor)));
    }
    {
        std::ifstream csvStrm("testfiles/bdYIMSp.csv");
        CPPUNIT_ASSERT(csvStrm.is_open());

        CTimeCheckingVisitor visitor("date",
                                     "%b %d %Y %I:%M:%S %p",
                                     expectedTimes);

        ml::api::CCsvInputParser parser(csvStrm);

        CPPUNIT_ASSERT(parser.readStream(std::ref(visitor)));
    }
    {
        std::ifstream csvStrm("testfiles/YmdHMS.csv");
        CPPUNIT_ASSERT(csvStrm.is_open());

        CTimeCheckingVisitor visitor("time",
                                     "%Y-%m-%d %H:%M:%S",
                                     expectedTimes);

        ml::api::CCsvInputParser parser(csvStrm);

        CPPUNIT_ASSERT(parser.readStream(std::ref(visitor)));
    }
    {
        std::ifstream csvStrm("testfiles/YmdHMSZ_GMT.csv");
        CPPUNIT_ASSERT(csvStrm.is_open());

        CTimeCheckingVisitor visitor("mytime",
                                     "%Y-%m-%d %H:%M:%S %Z",
                                     expectedTimes);

        ml::api::CCsvInputParser parser(csvStrm);

        CPPUNIT_ASSERT(parser.readStream(std::ref(visitor)));
    }

    // Switch to US Eastern time for this test
    CPPUNIT_ASSERT(ml::core::CTimezone::setTimezone("America/New_York"));

    {
        std::ifstream csvStrm("testfiles/YmdHMSZ_EST.csv");
        CPPUNIT_ASSERT(csvStrm.is_open());

        CTimeCheckingVisitor visitor("datetime",
                                     "%Y-%m-%d %H:%M:%S %Z",
                                     expectedTimes);

        ml::api::CCsvInputParser parser(csvStrm);

        CPPUNIT_ASSERT(parser.readStream(std::ref(visitor)));
    }

    // Set the timezone back to nothing, i.e. let the operating system decide
    // what to use
    CPPUNIT_ASSERT(ml::core::CTimezone::setTimezone(""));
}

void CCsvInputParserTest::testQuoteParsing(void) {
    // Expect:
    // q1 =
    // q2 =
    // q3 = "
    // q4 = ""
    std::string input(
        "b,q1,q2,q3,q4,e\n"
        "x,,\"\",\"\"\"\",\"\"\"\"\"\",x\n"
        );

    ml::api::CCsvInputParser parser(input);

    CQuoteCheckingVisitor visitor;

    CPPUNIT_ASSERT(parser.readStream(std::ref(visitor)));

    CPPUNIT_ASSERT_EQUAL(size_t(1), visitor.recordCount());
}

void CCsvInputParserTest::testLineParser(void) {
    ml::api::CCsvInputParser::CCsvLineParser lineParser;
    std::string token;

    {
        std::string simple{"a,b,c"};
        lineParser.reset(simple);

        CPPUNIT_ASSERT(!lineParser.atEnd());
        CPPUNIT_ASSERT(lineParser.parseNext(token));
        CPPUNIT_ASSERT_EQUAL(std::string("a"), token);

        CPPUNIT_ASSERT(!lineParser.atEnd());
        CPPUNIT_ASSERT(lineParser.parseNext(token));
        CPPUNIT_ASSERT_EQUAL(std::string("b"), token);

        CPPUNIT_ASSERT(!lineParser.atEnd());
        CPPUNIT_ASSERT(lineParser.parseNext(token));
        CPPUNIT_ASSERT_EQUAL(std::string("c"), token);

        CPPUNIT_ASSERT(lineParser.atEnd());
        CPPUNIT_ASSERT(!lineParser.parseNext(token));
    }
    {
        std::string quoted{"\"a,b,c\",b and some spaces,\"c quoted unecessarily\",\"d with a literal \"\"\""};
        lineParser.reset(quoted);

        CPPUNIT_ASSERT(!lineParser.atEnd());
        CPPUNIT_ASSERT(lineParser.parseNext(token));
        CPPUNIT_ASSERT_EQUAL(std::string("a,b,c"), token);

        CPPUNIT_ASSERT(!lineParser.atEnd());
        CPPUNIT_ASSERT(lineParser.parseNext(token));
        CPPUNIT_ASSERT_EQUAL(std::string("b and some spaces"), token);

        CPPUNIT_ASSERT(!lineParser.atEnd());
        CPPUNIT_ASSERT(lineParser.parseNext(token));
        CPPUNIT_ASSERT_EQUAL(std::string("c quoted unecessarily"), token);

        CPPUNIT_ASSERT(!lineParser.atEnd());
        CPPUNIT_ASSERT(lineParser.parseNext(token));
        CPPUNIT_ASSERT_EQUAL(std::string("d with a literal \""), token);

        CPPUNIT_ASSERT(lineParser.atEnd());
        CPPUNIT_ASSERT(!lineParser.parseNext(token));
    }
    {
        std::string cjk{"编码,コーディング,코딩"};
        lineParser.reset(cjk);

        CPPUNIT_ASSERT(!lineParser.atEnd());
        CPPUNIT_ASSERT(lineParser.parseNext(token));
        CPPUNIT_ASSERT_EQUAL(std::string("编码"), token);

        CPPUNIT_ASSERT(!lineParser.atEnd());
        CPPUNIT_ASSERT(lineParser.parseNext(token));
        CPPUNIT_ASSERT_EQUAL(std::string("コーディング"), token);

        CPPUNIT_ASSERT(!lineParser.atEnd());
        CPPUNIT_ASSERT(lineParser.parseNext(token));
        CPPUNIT_ASSERT_EQUAL(std::string("코딩"), token);

        CPPUNIT_ASSERT(lineParser.atEnd());
        CPPUNIT_ASSERT(!lineParser.parseNext(token));
    }
}

