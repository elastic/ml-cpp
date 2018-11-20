/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CLengthEncodedInputParserTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>
#include <core/CTimeUtils.h>

#include <api/CCsvInputParser.h>
#include <api/CLengthEncodedInputParser.h>

#include <fstream>
#include <functional>
#include <ios>
#include <sstream>

// For htonl
#ifdef Windows
#include <WinSock2.h>
#else
#include <netinet/in.h>
#endif

CppUnit::Test* CLengthEncodedInputParserTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CLengthEncodedInputParserTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CLengthEncodedInputParserTest>(
        "CLengthEncodedInputParserTest::testCsvEquivalence",
        &CLengthEncodedInputParserTest::testCsvEquivalence));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLengthEncodedInputParserTest>(
        "CLengthEncodedInputParserTest::testThroughput",
        &CLengthEncodedInputParserTest::testThroughput));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLengthEncodedInputParserTest>(
        "CLengthEncodedInputParserTest::testCorruptStreamDetection",
        &CLengthEncodedInputParserTest::testCorruptStreamDetection));

    return suiteOfTests;
}

namespace {

//! To save having binary files in the git repo, this class accepts records
//! from some text format and writes them to a temporary length encoded file
class CSetupVisitor {
public:
    CSetupVisitor() : m_RecordsPerBlock(0) {}

    //! Handle a record
    bool operator()(const ml::api::CCsvInputParser::TStrVec& fieldNames,
                    const ml::api::CCsvInputParser::TStrVec& fieldValues) {
        if (m_EncodedFieldNames.empty()) {
            this->appendNumber(fieldNames.size(), m_EncodedFieldNames);
            for (const auto& fieldName : fieldNames) {
                this->appendNumber(fieldName.length(), m_EncodedFieldNames);
                m_EncodedFieldNames += fieldName;
            }
        }

        this->appendNumber(fieldValues.size(), m_EncodedDataBlock);
        for (const auto& fieldValue : fieldValues) {
            this->appendNumber(fieldValue.length(), m_EncodedDataBlock);
            m_EncodedDataBlock += fieldValue;
        }

        ++m_RecordsPerBlock;

        return true;
    }

    std::string input(size_t testSize) const {
        std::string str;
        str.reserve(m_EncodedFieldNames.length() + testSize * m_EncodedDataBlock.length());

        // Assign like this to avoid GNU copy-on-write (which would defeat
        // the preceding reserve)
        str.assign(m_EncodedFieldNames, 0, m_EncodedFieldNames.length());

        // Duplicate the binary data according to the test size
        for (size_t count = 0; count < testSize; ++count) {
            str += m_EncodedDataBlock;
        }

        LOG_DEBUG(<< "Input size is " << str.length());

        return str;
    }

    size_t recordsPerBlock() const { return m_RecordsPerBlock; }

private:
    template<typename NUM_TYPE>
    void appendNumber(NUM_TYPE num, std::string& str) {
        uint32_t netNum(htonl(static_cast<uint32_t>(num)));
        str.append(reinterpret_cast<char*>(&netNum), sizeof(netNum));
    }

private:
    std::string m_EncodedFieldNames;
    size_t m_RecordsPerBlock;
    std::string m_EncodedDataBlock;
};

class CVisitor {
public:
    CVisitor() : m_Fast(true), m_RecordCount(0) {}

    CVisitor(const ml::api::CCsvInputParser::TStrVec& expectedFieldNames)
        : m_Fast(false), m_RecordCount(0), m_ExpectedFieldNames(expectedFieldNames) {}

    //! Reset the record count ready for another run
    void reset() { m_RecordCount = 0; }

    //! Handle a record in map form
    bool operator()(const ml::api::CLengthEncodedInputParser::TStrStrUMap& dataRowFields) {
        ++m_RecordCount;

        // For the throughput test, the assertions below will skew the
        // results, so bypass them
        if (m_Fast) {
            return true;
        }

        // Check the field names
        CPPUNIT_ASSERT_EQUAL(m_ExpectedFieldNames.size(), dataRowFields.size());
        for (ml::api::CCsvInputParser::TStrStrUMapCItr iter = dataRowFields.begin();
             iter != dataRowFields.end(); ++iter) {
            LOG_DEBUG(<< "Field " << iter->first << " is " << iter->second);
            CPPUNIT_ASSERT(std::find(m_ExpectedFieldNames.begin(),
                                     m_ExpectedFieldNames.end(),
                                     iter->first) != m_ExpectedFieldNames.end());
        }

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
}

void CLengthEncodedInputParserTest::testCsvEquivalence() {
    std::ifstream ifs("testfiles/simple.txt");
    CPPUNIT_ASSERT(ifs.is_open());

    CSetupVisitor setupVisitor;

    ml::api::CCsvInputParser setupParser(ifs);

    CPPUNIT_ASSERT(setupParser.readStreamIntoVecs(std::ref(setupVisitor)));

    // Input must be binary otherwise Windows will stop at CTRL+Z
    std::istringstream input(setupVisitor.input(1), std::ios::in | std::ios::binary);

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
    ml::api::CLengthEncodedInputParser parser1(input);
    CPPUNIT_ASSERT(parser1.readStreamIntoMaps(std::ref(visitor)));
    CPPUNIT_ASSERT_EQUAL(size_t(15), visitor.recordCount());

    // Now re-read to vectors
    ifs.clear();
    ifs.seekg(0);
    visitor.reset();

    ml::api::CCsvInputParser parser2(ifs);
    CPPUNIT_ASSERT(parser2.readStreamIntoVecs(std::ref(visitor)));
    CPPUNIT_ASSERT_EQUAL(size_t(15), visitor.recordCount());
}

void CLengthEncodedInputParserTest::testThroughput() {
    // NB: For fair comparison with the other input formats (CSV and Google
    // Protocol Buffers), the input data and test size must be identical

    LOG_DEBUG(<< "Creating throughput test data");

    std::ifstream ifs("testfiles/simple.txt");
    CPPUNIT_ASSERT(ifs.is_open());

    CSetupVisitor setupVisitor;

    ml::api::CCsvInputParser setupParser(ifs);

    CPPUNIT_ASSERT(setupParser.readStreamIntoVecs(std::ref(setupVisitor)));

    // Construct a large test input
    static const size_t TEST_SIZE(10000);
    // Input must be binary otherwise Windows will stop at CTRL+Z
    std::istringstream input(setupVisitor.input(TEST_SIZE), std::ios::in | std::ios::binary);

    ml::api::CLengthEncodedInputParser parser(input);

    CVisitor visitor;

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting throughput test at " << ml::core::CTimeUtils::toTimeString(start));

    CPPUNIT_ASSERT(parser.readStreamIntoMaps(std::ref(visitor)));

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished throughput test at " << ml::core::CTimeUtils::toTimeString(end));

    CPPUNIT_ASSERT_EQUAL(setupVisitor.recordsPerBlock() * TEST_SIZE, visitor.recordCount());

    LOG_INFO(<< "Parsing " << visitor.recordCount() << " records took "
             << (end - start) << " seconds");
}

void CLengthEncodedInputParserTest::testCorruptStreamDetection() {
    uint32_t numFields(1);
    uint32_t numFieldsNet(htonl(numFields));
    std::string dodgyInput(reinterpret_cast<char*>(&numFieldsNet), sizeof(uint32_t));
    // This is going to create a length field consisting of four 'a' characters
    // interpreted as a uint32_t
    dodgyInput.append(1000, 'a');

    // Input must be binary otherwise Windows will stop at CTRL+Z
    std::istringstream input(dodgyInput, std::ios::in | std::ios::binary);

    ml::api::CLengthEncodedInputParser parser(input);

    CVisitor visitor;

    LOG_INFO(<< "Expect the next parse to report a suspiciously long length");
    CPPUNIT_ASSERT(!parser.readStreamIntoMaps(std::ref(visitor)));
}
