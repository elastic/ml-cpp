/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>
#include <core/CTimeUtils.h>
#include <core/CTimezone.h>
#include <core/CoreTypes.h>

#include <api/CCsvInputParser.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <fstream>
#include <functional>
#include <vector>

BOOST_TEST_DONT_PRINT_LOG_VALUE(ml::api::CCsvInputParser::TStrVecItr)
BOOST_TEST_DONT_PRINT_LOG_VALUE(ml::api::CCsvInputParser::TStrVecCItr)
BOOST_TEST_DONT_PRINT_LOG_VALUE(ml::api::CCsvInputParser::TStrStrUMapCItr)

BOOST_AUTO_TEST_SUITE(CCsvInputParserTest)

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
            BOOST_TEST_REQUIRE(iter != m_ExpectedFieldNames.end());
        }

        BOOST_REQUIRE_EQUAL(m_ExpectedFieldNames.size(), dataRowFields.size());

        // Check the line count is consistent with the _raw field
        ml::api::CCsvInputParser::TStrStrUMapCItr rawIter = dataRowFields.find("_raw");
        BOOST_TEST_REQUIRE(rawIter != dataRowFields.end());
        ml::api::CCsvInputParser::TStrStrUMapCItr lineCountIter =
            dataRowFields.find("linecount");
        BOOST_TEST_REQUIRE(lineCountIter != dataRowFields.end());

        size_t expectedLineCount(1 + std::count(rawIter->second.begin(),
                                                rawIter->second.end(), '\n'));
        size_t lineCount(0);
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType(lineCountIter->second, lineCount));
        BOOST_REQUIRE_EQUAL(expectedLineCount, lineCount);

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
        BOOST_REQUIRE_EQUAL(ml::core::CContainerPrinter::print(m_ExpectedFieldNames),
                            ml::core::CContainerPrinter::print(fieldNames));

        BOOST_REQUIRE_EQUAL(m_ExpectedFieldNames.size(), fieldValues.size());

        // Check the line count is consistent with the _raw field
        auto rawIter = std::find(fieldNames.begin(), fieldNames.end(), "_raw");
        BOOST_TEST_REQUIRE(rawIter != fieldNames.end());
        auto lineCountIter = std::find(fieldNames.begin(), fieldNames.end(), "linecount");
        BOOST_TEST_REQUIRE(lineCountIter != fieldNames.end());

        const std::string& rawStr = fieldValues[rawIter - fieldNames.begin()];
        std::size_t expectedLineCount(1 + std::count(rawStr.begin(), rawStr.end(), '\n'));
        std::size_t lineCount(0);
        const std::string& lineCountStr = fieldValues[lineCountIter - fieldNames.begin()];
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType(lineCountStr, lineCount));
        BOOST_REQUIRE_EQUAL(expectedLineCount, lineCount);

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
        BOOST_TEST_REQUIRE(m_RecordCount < m_ExpectedTimes.size());

        auto iter = dataRowFields.find(m_TimeField);
        BOOST_TEST_REQUIRE(iter != dataRowFields.end());

        // Now check the actual time
        ml::api::CCsvInputParser::TStrStrUMapCItr fieldIter = dataRowFields.find(m_TimeField);
        BOOST_TEST_REQUIRE(fieldIter != dataRowFields.end());
        ml::core_t::TTime timeVal(0);
        if (m_TimeFormat.empty()) {
            BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType(fieldIter->second, timeVal));
        } else {
            BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(
                m_TimeFormat, fieldIter->second, timeVal));
            LOG_DEBUG(<< "Converted " << fieldIter->second << " to " << timeVal
                      << " using format " << m_TimeFormat);
        }
        BOOST_REQUIRE_EQUAL(m_ExpectedTimes[m_RecordCount], timeVal);

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
        BOOST_TEST_REQUIRE(fieldIter != dataRowFields.end());
        BOOST_REQUIRE_EQUAL(std::string(""), fieldIter->second);

        fieldIter = dataRowFields.find("q2");
        BOOST_TEST_REQUIRE(fieldIter != dataRowFields.end());
        BOOST_REQUIRE_EQUAL(std::string(""), fieldIter->second);

        fieldIter = dataRowFields.find("q3");
        BOOST_TEST_REQUIRE(fieldIter != dataRowFields.end());
        BOOST_REQUIRE_EQUAL(std::string("\""), fieldIter->second);

        fieldIter = dataRowFields.find("q4");
        BOOST_TEST_REQUIRE(fieldIter != dataRowFields.end());
        BOOST_REQUIRE_EQUAL(std::string("\"\""), fieldIter->second);

        ++m_RecordCount;

        return true;
    }

    size_t recordCount() const { return m_RecordCount; }

private:
    size_t m_RecordCount;
};
}

BOOST_AUTO_TEST_CASE(testSimpleDelims) {
    std::ifstream simpleStrm("testfiles/simple.txt");
    BOOST_TEST_REQUIRE(simpleStrm.is_open());

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
    BOOST_TEST_REQUIRE(parser1.readStreamIntoMaps(std::ref(visitor)));
    BOOST_REQUIRE_EQUAL(size_t(15), visitor.recordCount());

    // Now re-read to vectors
    simpleStrm.clear();
    simpleStrm.seekg(0);
    visitor.reset();

    ml::api::CCsvInputParser parser2(simpleStrm);
    BOOST_TEST_REQUIRE(parser2.readStreamIntoVecs(std::ref(visitor)));
    BOOST_REQUIRE_EQUAL(size_t(15), visitor.recordCount());
}

BOOST_AUTO_TEST_CASE(testComplexDelims) {
    std::ifstream complexStrm("testfiles/complex.txt");
    BOOST_TEST_REQUIRE(complexStrm.is_open());

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
    BOOST_TEST_REQUIRE(parser1.readStreamIntoMaps(std::ref(visitor)));

    // Now re-read to vectors
    complexStrm.clear();
    complexStrm.seekg(0);
    visitor.reset();

    ml::api::CCsvInputParser parser2(complexStrm);
    BOOST_TEST_REQUIRE(parser2.readStreamIntoVecs(std::ref(visitor)));
}

BOOST_AUTO_TEST_CASE(testThroughput) {
    std::ifstream ifs("testfiles/simple.txt");
    BOOST_TEST_REQUIRE(ifs.is_open());

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
    BOOST_TEST_REQUIRE((nonHeaderLines % 2) == 0);
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

    BOOST_TEST_REQUIRE(parser.readStreamIntoMaps(std::ref(visitor)));

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished throughput test at " << ml::core::CTimeUtils::toTimeString(end));

    BOOST_REQUIRE_EQUAL(recordsPerBlock * TEST_SIZE, visitor.recordCount());

    LOG_INFO(<< "Parsing " << visitor.recordCount() << " records took "
             << (end - start) << " seconds");
}

BOOST_AUTO_TEST_CASE(testDateParse) {
    static const ml::core_t::TTime EXPECTED_TIMES[] = {
        1359331200, 1359331200, 1359331207, 1359331220, 1359331259, 1359331262,
        1359331269, 1359331270, 1359331272, 1359331296, 1359331301, 1359331311,
        1359331314, 1359331315, 1359331316, 1359331321, 1359331328, 1359331333,
        1359331349, 1359331352, 1359331370, 1359331382, 1359331385, 1359331386,
        1359331395, 1359331404, 1359331416, 1359331416, 1359331424, 1359331429};

    CTimeCheckingVisitor::TTimeVec expectedTimes(std::begin(EXPECTED_TIMES),
                                                 std::end(EXPECTED_TIMES));

    // Ensure we are in UK timewise
    BOOST_TEST_REQUIRE(ml::core::CTimezone::setTimezone("Europe/London"));

    {
        std::ifstream csvStrm("testfiles/s.csv");
        BOOST_TEST_REQUIRE(csvStrm.is_open());

        CTimeCheckingVisitor visitor("time", "", expectedTimes);

        ml::api::CCsvInputParser parser(csvStrm);

        BOOST_TEST_REQUIRE(parser.readStreamIntoMaps(std::ref(visitor)));
    }
    {
        std::ifstream csvStrm("testfiles/bdYIMSp.csv");
        BOOST_TEST_REQUIRE(csvStrm.is_open());

        CTimeCheckingVisitor visitor("date", "%b %d %Y %I:%M:%S %p", expectedTimes);

        ml::api::CCsvInputParser parser(csvStrm);

        BOOST_TEST_REQUIRE(parser.readStreamIntoMaps(std::ref(visitor)));
    }
    {
        std::ifstream csvStrm("testfiles/YmdHMS.csv");
        BOOST_TEST_REQUIRE(csvStrm.is_open());

        CTimeCheckingVisitor visitor("time", "%Y-%m-%d %H:%M:%S", expectedTimes);

        ml::api::CCsvInputParser parser(csvStrm);

        BOOST_TEST_REQUIRE(parser.readStreamIntoMaps(std::ref(visitor)));
    }
    {
        std::ifstream csvStrm("testfiles/YmdHMSZ_GMT.csv");
        BOOST_TEST_REQUIRE(csvStrm.is_open());

        CTimeCheckingVisitor visitor("mytime", "%Y-%m-%d %H:%M:%S %Z", expectedTimes);

        ml::api::CCsvInputParser parser(csvStrm);

        BOOST_TEST_REQUIRE(parser.readStreamIntoMaps(std::ref(visitor)));
    }

    // Switch to US Eastern time for this test
    BOOST_TEST_REQUIRE(ml::core::CTimezone::setTimezone("America/New_York"));

    {
        std::ifstream csvStrm("testfiles/YmdHMSZ_EST.csv");
        BOOST_TEST_REQUIRE(csvStrm.is_open());

        CTimeCheckingVisitor visitor("datetime", "%Y-%m-%d %H:%M:%S %Z", expectedTimes);

        ml::api::CCsvInputParser parser(csvStrm);

        BOOST_TEST_REQUIRE(parser.readStreamIntoMaps(std::ref(visitor)));
    }

    // Set the timezone back to nothing, i.e. let the operating system decide
    // what to use
    BOOST_TEST_REQUIRE(ml::core::CTimezone::setTimezone(""));
}

BOOST_AUTO_TEST_CASE(testQuoteParsing) {
    // Expect:
    // q1 =
    // q2 =
    // q3 = "
    // q4 = ""
    std::string input("b,q1,q2,q3,q4,e\n"
                      "x,,\"\",\"\"\"\",\"\"\"\"\"\",x\n");

    ml::api::CCsvInputParser parser(input);

    CQuoteCheckingVisitor visitor;

    BOOST_TEST_REQUIRE(parser.readStreamIntoMaps(std::ref(visitor)));

    BOOST_REQUIRE_EQUAL(size_t(1), visitor.recordCount());
}

BOOST_AUTO_TEST_SUITE_END()
