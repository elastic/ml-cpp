/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CNdJsonInputParserTest.h"

#include <core/CLogger.h>
#include <core/CTimeUtils.h>

#include <api/CCsvInputParser.h>
#include <api/CNdJsonInputParser.h>
#include <api/CNdJsonOutputWriter.h>

#include <fstream>
#include <functional>
#include <sstream>

CppUnit::Test* CNdJsonInputParserTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CNdJsonInputParserTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CNdJsonInputParserTest>(
        "CNdJsonInputParserTest::testThroughputArbitraryMapHandler",
        &CNdJsonInputParserTest::testThroughputArbitraryMapHandler));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNdJsonInputParserTest>(
        "CNdJsonInputParserTest::testThroughputCommonMapHandler",
        &CNdJsonInputParserTest::testThroughputCommonMapHandler));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNdJsonInputParserTest>(
        "CNdJsonInputParserTest::testThroughputArbitraryVecHandler",
        &CNdJsonInputParserTest::testThroughputArbitraryVecHandler));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNdJsonInputParserTest>(
        "CNdJsonInputParserTest::testThroughputCommonVecHandler",
        &CNdJsonInputParserTest::testThroughputCommonVecHandler));

    return suiteOfTests;
}

namespace {

class CSetupVisitor {
public:
    CSetupVisitor() : m_RecordsPerBlock(0) {}

    //! Handle a record
    bool operator()(const ml::api::CCsvInputParser::TStrStrUMap& dataRowFields) {
        ++m_RecordsPerBlock;

        CPPUNIT_ASSERT(m_OutputWriter.writeRow(dataRowFields));

        return true;
    }

    std::string input(size_t testSize) const {
        const std::string& block = m_OutputWriter.internalString();

        std::string str;
        str.reserve(testSize * block.length());

        // Duplicate the binary data according to the test size
        for (size_t count = 0; count < testSize; ++count) {
            str.append(block);
        }

        LOG_DEBUG(<< "Input size is " << str.length());

        return str;
    }

    size_t recordsPerBlock() const { return m_RecordsPerBlock; }

private:
    size_t m_RecordsPerBlock;
    ml::api::CNdJsonOutputWriter m_OutputWriter;
};

class CVisitor {
public:
    CVisitor() : m_RecordCount(0) {}

    //! Handle a record in map form
    bool operator()(const ml::api::CNdJsonInputParser::TStrStrUMap& /*dataRowFields*/) {
        ++m_RecordCount;
        return true;
    }

    //! Handle a record in vector form
    bool operator()(const ml::api::CNdJsonInputParser::TStrVec& /*fieldNames*/,
                    const ml::api::CNdJsonInputParser::TStrVec& /*fieldValues*/) {
        ++m_RecordCount;
        return true;
    }

    size_t recordCount() const { return m_RecordCount; }

private:
    size_t m_RecordCount;
};
}

void CNdJsonInputParserTest::testThroughputArbitraryMapHandler() {
    LOG_INFO(<< "Testing parse to map assuming arbitrary fields in JSON documents");
    this->runTest(false, false);
}

void CNdJsonInputParserTest::testThroughputCommonMapHandler() {
    LOG_INFO(<< "Testing parse to map assuming all JSON documents have the same fields");
    this->runTest(true, false);
}

void CNdJsonInputParserTest::testThroughputArbitraryVecHandler() {
    LOG_INFO(<< "Testing parse to vectors assuming arbitrary fields in JSON documents");
    this->runTest(false, true);
}

void CNdJsonInputParserTest::testThroughputCommonVecHandler() {
    LOG_INFO(<< "Testing parse to vectors assuming all JSON documents have the same fields");
    this->runTest(true, true);
}

void CNdJsonInputParserTest::runTest(bool allDocsSameStructure, bool parseAsVecs) {
    // NB: For fair comparison with the other input formats (CSV and Google
    // Protocol Buffers), the input data and test size must be identical

    LOG_DEBUG(<< "Creating throughput test data");

    std::ifstream ifs("testfiles/simple.txt");
    CPPUNIT_ASSERT(ifs.is_open());

    CSetupVisitor setupVisitor;

    ml::api::CCsvInputParser setupParser(ifs);

    CPPUNIT_ASSERT(setupParser.readStreamAsMaps(std::ref(setupVisitor)));

    // Construct a large test input
    static const size_t TEST_SIZE(5000);
    std::istringstream input(setupVisitor.input(TEST_SIZE));

    ml::api::CNdJsonInputParser parser(input, allDocsSameStructure);

    CVisitor visitor;

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Starting throughput test at " << ml::core::CTimeUtils::toTimeString(start));

    if (parseAsVecs) {
        CPPUNIT_ASSERT(parser.readStreamAsVecs(std::ref(visitor)));
    } else {
        CPPUNIT_ASSERT(parser.readStreamAsMaps(std::ref(visitor)));
    }

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO(<< "Finished throughput test at " << ml::core::CTimeUtils::toTimeString(end));

    CPPUNIT_ASSERT_EQUAL(setupVisitor.recordsPerBlock() * TEST_SIZE, visitor.recordCount());

    LOG_INFO(<< "Parsing " << visitor.recordCount() << " records took "
             << (end - start) << " seconds");
}
