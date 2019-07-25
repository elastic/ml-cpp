/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CCsvInputParserTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>
#include <core/CTimeUtils.h>
#include <core/CTimezone.h>
#include <core/CoreTypes.h>

#include <api/CCsvInputParser.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <vector>

CppUnit::Test* CCsvInputParserTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CCsvInputParserTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CCsvInputParserTest>(
        "CCsvInputParserTest::testSimpleDelims", &CCsvInputParserTest::testSimpleDelims));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCsvInputParserTest>(
        "CCsvInputParserTest::testComplexDelims", &CCsvInputParserTest::testComplexDelims));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCsvInputParserTest>(
        "CCsvInputParserTest::testThroughput", &CCsvInputParserTest::testThroughput));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCsvInputParserTest>(
        "CCsvInputParserTest::testDateParse", &CCsvInputParserTest::testDateParse));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCsvInputParserTest>(
        "CCsvInputParserTest::testQuoteParsing", &CCsvInputParserTest::testQuoteParsing));
    suiteOfTests->addTest(new CppUnit::TestCaller<CCsvInputParserTest>(
        "CCsvInputParserTest::testLineParser", &CCsvInputParserTest::testLineParser));

    return suiteOfTests;
}

namespace {

class CVisitor {
public:
    CVisitor() : m_Fast(true), m_RecordCount(0) {}

    CVisitor(const ml::api::CCsvInputParser::TStrVec& expectedFieldNames)
        : m_Fast(false), m_RecordCount(0), m_ExpectedFieldNames(expectedFieldNames) {}

    //! Reset the record count ready for another run
    void reset() { m_RecordCount = 0; }

    //! Handle a record in map form
    bool operator()(const ml::api::CCsvInputParser::TStrStrUMap& dataRowFields) {
        ++m_RecordCount;

        // For the throughput test, the assertions below will skew the
        // results, so bypass them
        if (m_Fast) {
            return true;
        }

        // Check the field names
        for (const auto& entry : dataRowFields) {
            auto iter = std::find(m_ExpectedFieldNames.begin(),
                                  m_ExpectedFieldNames.end(), entry.first);
            CPPUNIT_ASSERT(iter != m_ExpectedFieldNames.end());
        }

        CPPUNIT_ASSERT_EQUAL(m_ExpectedFieldNames.size(), dataRowFields.size());

        // Check the line count is consistent with the _raw field
        ml::api::CCsvInputParser::TStrStrUMapCItr rawIter = dataRowFields.find("_raw");
        CPPUNIT_ASSERT(rawIter != dataRowFields.end());
        ml::api::CCsvInputParser::TStrStrUMapCItr lineCountIter =
            dataRowFields.find("linecount");
        CPPUNIT_ASSERT(lineCountIter != dataRowFields.end());

        size_t expectedLineCount(1 + std::count(rawIter->second.begin(),
                                                rawIter->second.end(), '\n'));
        size_t lineCount(0);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType(lineCountIter->second, lineCount));
        CPPUNIT_ASSERT_EQUAL(expectedLineCount, lineCount);

        return true;
    }

    //! Handle a record in vector form
    bool operator()(const ml::api::CCsvInputParser::TStrVec& fieldNames,
                    const ml::api::CCsvInputParser::TStrVec& fieldValues) {
        ++m_RecordCount;

        // For the throughput test, the assertions below will skew the
        // results, so bypass them
        if (m_Fast) {
            return true;
        }

        // Check the field names
        CPPUNIT_ASSERT_EQUAL(ml::core::CContainerPrinter::print(m_ExpectedFieldNames),
                             ml::core::CContainerPrinter::print(fieldNames));

        CPPUNIT_ASSERT_EQUAL(m_ExpectedFieldNames.size(), fieldValues.size());

        // Check the line count is consistent with the _raw field
        auto rawIter = std::find(fieldNames.begin(), fieldNames.end(), "_raw");
        CPPUNIT_ASSERT(rawIter != fieldNames.end());
        auto lineCountIter = std::find(fieldNames.begin(), fieldNames.end(), "linecount");
        CPPUNIT_ASSERT(lineCountIter != fieldNames.end());

        const std::string& rawStr = fieldValues[rawIter - fieldNames.begin()];
        std::size_t expectedLineCount(1 + std::count(rawStr.begin(), rawStr.end(), '\n'));
        std::size_t lineCount(0);
        const std::string& lineCountStr = fieldValues[lineCountIter - fieldNames.begin()];
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType(lineCountStr, lineCount));
        CPPUNIT_ASSERT_EQUAL(expectedLineCount, lineCount);

        return true;
    }

    size_t recordCount() const { return m_RecordCount; }

private:
    bool m_Fast;
    size_t m_RecordCount;
    ml::api::CCsvInputParser::TStrVec m_ExpectedFieldNames;
};

class CTimeCheckingVisitor {
public:
    using TTimeVec = std::vector<ml::core_t::TTime>;

public:
    CTimeCheckingVisitor(const std::string& timeField,
                         const std::string& timeFormat,
                         const TTimeVec& expectedTimes)
        : m_RecordCount(0), m_TimeField(timeField), m_TimeFormat(timeFormat),
          m_ExpectedTimes(expectedTimes) {}

    //! Handle a record
    bool operator()(const ml::api::CCsvInputParser::TStrStrUMap& dataRowFields) {
        // Check the time field exists
        CPPUNIT_ASSERT(m_RecordCount < m_ExpectedTimes.size());

        auto iter = dataRowFields.find(m_TimeField);
        CPPUNIT_ASSERT(iter != dataRowFields.end());

        // Now check the actual time
        ml::api::CCsvInputParser::TStrStrUMapCItr fieldIter = dataRowFields.find(m_TimeField);
        CPPUNIT_ASSERT(fieldIter != dataRowFields.end());
        ml::core_t::TTime timeVal(0);
        if (m_TimeFormat.empty()) {
            CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType(fieldIter->second, timeVal));
        } else {
            CPPUNIT_ASSERT(ml::core::CTimeUtils::strptime(
                m_TimeFormat, fieldIter->second, timeVal));
            LOG_DEBUG(<< "Converted " << fieldIter->second << " to " << timeVal
                      << " using format " << m_TimeFormat);
        }
        CPPUNIT_ASSERT_EQUAL(m_ExpectedTimes[m_RecordCount], timeVal);

        ++m_RecordCount;

        return true;
    }

    size_t recordCount() const { return m_RecordCount; }

private:
    size_t m_RecordCount;
    std::string m_TimeField;
    std::string m_TimeFormat;
    TTimeVec m_ExpectedTimes;
};

class CQuoteCheckingVisitor {
public:
    CQuoteCheckingVisitor() : m_RecordCount(0) {}

    //! Handle a record
    bool operator()(const ml::api::CCsvInputParser::TStrStrUMap& dataRowFields) {
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

    size_t recordCount() const { return m_RecordCount; }

private:
    size_t m_RecordCount;
};
}

void CCsvInputParserTest::testSimpleDelims() {
    std::ifstream simpleStrm("testfiles/simple.txt");
    CPPUNIT_ASSERT(simpleStrm.is_open());

    ml::api::CCsvInputParser::TStrVec expectedFieldNames;
    expectedFieldNames.emplace_back("_cd");
    expectedFieldNames.emplace_back("_indextime");
    expectedFieldNames.emplace_back("_kv");
    expectedFieldNames.emplace_back("_raw");
    expectedFieldNames.emplace_back("_serial");
    expectedFieldNames.emplace_back("_si");
    expectedFieldNames.emplace_back("_sourcetype");
    expectedFieldNames.emplace_back("_time");
    expectedFieldNames.emplace_back("date_hour");
    expectedFieldNames.emplace_back("date_mday");
    expectedFieldNames.emplace_back("date_minute");
    expectedFieldNames.emplace_back("date_month");
    expectedFieldNames.emplace_back("date_second");
    expectedFieldNames.emplace_back("date_wday");
    expectedFieldNames.emplace_back("date_year");
    expectedFieldNames.emplace_back("date_zone");
    expectedFieldNames.emplace_back("eventtype");
    expectedFieldNames.emplace_back("host");
    expectedFieldNames.emplace_back("index");
    expectedFieldNames.emplace_back("linecount");
    expectedFieldNames.emplace_back("punct");
    expectedFieldNames.emplace_back("source");
    expectedFieldNames.emplace_back("sourcetype");
    expectedFieldNames.emplace_back("server");
    expectedFieldNames.emplace_back("timeendpos");
    expectedFieldNames.emplace_back("timestartpos");

    CVisitor visitor(expectedFieldNames);

    // First read to a map
    ml::api::CCsvInputParser parser1(simpleStrm);
    CPPUNIT_ASSERT(parser1.readStreamIntoMaps(std::ref(visitor)));
    CPPUNIT_ASSERT_EQUAL(size_t(15), visitor.recordCount());

    // Now re-read to vectors
    simpleStrm.clear();
    simpleStrm.seekg(0);
    visitor.reset();

    ml::api::CCsvInputParser parser2(simpleStrm);
    CPPUNIT_ASSERT(parser2.readStreamIntoVecs(std::ref(visitor)));
    CPPUNIT_ASSERT_EQUAL(size_t(15), visitor.recordCount());
}

void CCsvInputParserTest::testComplexDelims() {
    std::ifstream complexStrm("testfiles/complex.txt");
    CPPUNIT_ASSERT(complexStrm.is_open());

    ml::api::CCsvInputParser::TStrVec expectedFieldNames;
    expectedFieldNames.emplace_back("_cd");
    expectedFieldNames.emplace_back("_indextime");
    expectedFieldNames.emplace_back("_kv");
    expectedFieldNames.emplace_back("_raw");
    expectedFieldNames.emplace_back("_serial");
    expectedFieldNames.emplace_back("_si");
    expectedFieldNames.emplace_back("_sourcetype");
    expectedFieldNames.emplace_back("_time");
    expectedFieldNames.emplace_back("date_hour");
    expectedFieldNames.emplace_back("date_mday");
    expectedFieldNames.emplace_back("date_minute");
    expectedFieldNames.emplace_back("date_month");
    expectedFieldNames.emplace_back("date_second");
    expectedFieldNames.emplace_back("date_wday");
    expectedFieldNames.emplace_back("date_year");
    expectedFieldNames.emplace_back("date_zone");
    expectedFieldNames.emplace_back("eventtype");
    expectedFieldNames.emplace_back("host");
    expectedFieldNames.emplace_back("index");
    expectedFieldNames.emplace_back("linecount");
    expectedFieldNames.emplace_back("punct");
    expectedFieldNames.emplace_back("source");
    expectedFieldNames.emplace_back("sourcetype");
    expectedFieldNames.emplace_back("server");
    expectedFieldNames.emplace_back("timeendpos");
    expectedFieldNames.emplace_back("timestartpos");

    CVisitor visitor(expectedFieldNames);

    // First read to a map
    ml::api::CCsvInputParser parser1(complexStrm);
    CPPUNIT_ASSERT(parser1.readStreamIntoMaps(std::ref(visitor)));

    // Now re-read to vectors
    complexStrm.clear();
    complexStrm.seekg(0);
    visitor.reset();

    ml::api::CCsvInputParser parser2(complexStrm);
    CPPUNIT_ASSERT(parser2.readStreamIntoVecs(std::ref(visitor)));
}

void CCsvInputParserTest::testThroughput() {
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
    LOG_DEBUG(<< "Input size is " << input.length());

    ml::api::CCsvInputParser parser(input);

    CVisitor visitor;

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting throughput test at " << ml::core::CTimeUtils::toTimeString(start));

    CPPUNIT_ASSERT(parser.readStreamIntoMaps(std::ref(visitor)));

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished throughput test at " << ml::core::CTimeUtils::toTimeString(end));

    CPPUNIT_ASSERT_EQUAL(recordsPerBlock * TEST_SIZE, visitor.recordCount());

    LOG_INFO(<< "Parsing " << visitor.recordCount() << " records took "
             << (end - start) << " seconds");
}

void CCsvInputParserTest::testDateParse() {
    static const ml::core_t::TTime EXPECTED_TIMES[] = {
        1359331200, 1359331200, 1359331207, 1359331220, 1359331259, 1359331262,
        1359331269, 1359331270, 1359331272, 1359331296, 1359331301, 1359331311,
        1359331314, 1359331315, 1359331316, 1359331321, 1359331328, 1359331333,
        1359331349, 1359331352, 1359331370, 1359331382, 1359331385, 1359331386,
        1359331395, 1359331404, 1359331416, 1359331416, 1359331424, 1359331429};

    CTimeCheckingVisitor::TTimeVec expectedTimes(std::begin(EXPECTED_TIMES),
                                                 std::end(EXPECTED_TIMES));

    // Ensure we are in UK timewise
    CPPUNIT_ASSERT(ml::core::CTimezone::setTimezone("Europe/London"));

    {
        std::ifstream csvStrm("testfiles/s.csv");
        CPPUNIT_ASSERT(csvStrm.is_open());

        CTimeCheckingVisitor visitor("time", "", expectedTimes);

        ml::api::CCsvInputParser parser(csvStrm);

        CPPUNIT_ASSERT(parser.readStreamIntoMaps(std::ref(visitor)));
    }
    {
        std::ifstream csvStrm("testfiles/bdYIMSp.csv");
        CPPUNIT_ASSERT(csvStrm.is_open());

        CTimeCheckingVisitor visitor("date", "%b %d %Y %I:%M:%S %p", expectedTimes);

        ml::api::CCsvInputParser parser(csvStrm);

        CPPUNIT_ASSERT(parser.readStreamIntoMaps(std::ref(visitor)));
    }
    {
        std::ifstream csvStrm("testfiles/YmdHMS.csv");
        CPPUNIT_ASSERT(csvStrm.is_open());

        CTimeCheckingVisitor visitor("time", "%Y-%m-%d %H:%M:%S", expectedTimes);

        ml::api::CCsvInputParser parser(csvStrm);

        CPPUNIT_ASSERT(parser.readStreamIntoMaps(std::ref(visitor)));
    }
    {
        std::ifstream csvStrm("testfiles/YmdHMSZ_GMT.csv");
        CPPUNIT_ASSERT(csvStrm.is_open());

        CTimeCheckingVisitor visitor("mytime", "%Y-%m-%d %H:%M:%S %Z", expectedTimes);

        ml::api::CCsvInputParser parser(csvStrm);

        CPPUNIT_ASSERT(parser.readStreamIntoMaps(std::ref(visitor)));
    }

    // Switch to US Eastern time for this test
    CPPUNIT_ASSERT(ml::core::CTimezone::setTimezone("America/New_York"));

    {
        std::ifstream csvStrm("testfiles/YmdHMSZ_EST.csv");
        CPPUNIT_ASSERT(csvStrm.is_open());

        CTimeCheckingVisitor visitor("datetime", "%Y-%m-%d %H:%M:%S %Z", expectedTimes);

        ml::api::CCsvInputParser parser(csvStrm);

        CPPUNIT_ASSERT(parser.readStreamIntoMaps(std::ref(visitor)));
    }

    // Set the timezone back to nothing, i.e. let the operating system decide
    // what to use
    CPPUNIT_ASSERT(ml::core::CTimezone::setTimezone(""));
}

void CCsvInputParserTest::testQuoteParsing() {
    // Expect:
    // q1 =
    // q2 =
    // q3 = "
    // q4 = ""
    std::string input("b,q1,q2,q3,q4,e\n"
                      "x,,\"\",\"\"\"\",\"\"\"\"\"\",x\n");

    ml::api::CCsvInputParser parser(input);

    CQuoteCheckingVisitor visitor;

    CPPUNIT_ASSERT(parser.readStreamIntoMaps(std::ref(visitor)));

    CPPUNIT_ASSERT_EQUAL(size_t(1), visitor.recordCount());
}

void CCsvInputParserTest::testLineParser() {
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
