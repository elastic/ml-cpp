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
#include "CLineifiedXmlInputParserTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CTimeUtils.h>
#include <core/CXmlParser.h>

#include <api/CCsvInputParser.h>
#include <api/CLineifiedXmlInputParser.h>
#include <api/CLineifiedXmlOutputWriter.h>

#include <functional>
#include <fstream>
#include <sstream>


CppUnit::Test *CLineifiedXmlInputParserTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CLineifiedXmlInputParserTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CLineifiedXmlInputParserTest>(
                                   "CLineifiedXmlInputParserTest::testThroughputArbitraryConformant",
                                   &CLineifiedXmlInputParserTest::testThroughputArbitraryConformant) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLineifiedXmlInputParserTest>(
                                   "CLineifiedXmlInputParserTest::testThroughputCommonConformant",
                                   &CLineifiedXmlInputParserTest::testThroughputCommonConformant) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLineifiedXmlInputParserTest>(
                                   "CLineifiedXmlInputParserTest::testThroughputArbitraryRapid",
                                   &CLineifiedXmlInputParserTest::testThroughputArbitraryRapid) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLineifiedXmlInputParserTest>(
                                   "CLineifiedXmlInputParserTest::testThroughputCommonRapid",
                                   &CLineifiedXmlInputParserTest::testThroughputCommonRapid) );

    return suiteOfTests;
}

namespace
{


class CSetupVisitor
{
    public:
        CSetupVisitor()
            : m_RecordsPerBlock(0),
              m_OutputWriter("root")
        {
        }

        //! Handle a record
        bool operator()(const ml::api::CCsvInputParser::TStrStrUMap &dataRowFields)
        {
            ++m_RecordsPerBlock;
            CPPUNIT_ASSERT(m_OutputWriter.writeRow(dataRowFields));

            return true;
        }

        std::string input(size_t testSize) const
        {
            const std::string &block = m_OutputWriter.internalString();

            std::string str;
            str.reserve(testSize * block.length());

            // Duplicate the binary data according to the test size
            for (size_t count = 0; count < testSize; ++count)
            {
                str.append(block);
            }

            LOG_DEBUG("Input size is " << str.length());

            return str;
        }

        size_t recordsPerBlock() const
        {
            return m_RecordsPerBlock;
        }

    private:
        size_t                                  m_RecordsPerBlock;
        ml::api::CLineifiedXmlOutputWriter m_OutputWriter;
};

class CVisitor
{
    public:
        CVisitor()
            : m_RecordCount(0)
        {
        }

        //! Handle a record
        bool operator()(const ml::api::CLineifiedXmlInputParser::TStrStrUMap &/*dataRowFields*/)
        {
            ++m_RecordCount;
            return true;
        }

        size_t recordCount() const
        {
            return m_RecordCount;
        }

    private:
        size_t m_RecordCount;
};


}

void CLineifiedXmlInputParserTest::testThroughputArbitraryConformant()
{
    LOG_INFO("Testing using a standards-conformant XML parser assuming arbitrary fields in XML documents");
    this->runTest<ml::core::CXmlParser>(false);
}

void CLineifiedXmlInputParserTest::testThroughputCommonConformant()
{
    LOG_INFO("Testing using a standards-conformant XML parser assuming all XML documents have the same fields");
    this->runTest<ml::core::CXmlParser>(true);
}

void CLineifiedXmlInputParserTest::testThroughputArbitraryRapid()
{
    LOG_INFO("Testing using a rapid XML parser assuming arbitrary fields in XML documents");
    this->runTest<ml::core::CRapidXmlParser>(false);
}

void CLineifiedXmlInputParserTest::testThroughputCommonRapid()
{
    LOG_INFO("Testing using a rapid XML parser assuming all XML documents have the same fields");
    this->runTest<ml::core::CRapidXmlParser>(true);
}

template <typename PARSER>
void CLineifiedXmlInputParserTest::runTest(bool allDocsSameStructure)
{
    // NB: For fair comparison with the other input formats (CSV and Google
    // Protocol Buffers), the input data and test size must be identical

    LOG_DEBUG("Creating throughput test data");

    std::ifstream ifs("testfiles/simple.txt");
    CPPUNIT_ASSERT(ifs.is_open());

    CSetupVisitor setupVisitor;

    ml::api::CCsvInputParser setupParser(ifs);

    CPPUNIT_ASSERT(setupParser.readStream(std::ref(setupVisitor)));

    // Construct a large test input
    static const size_t TEST_SIZE(5000);
    std::istringstream input(setupVisitor.input(TEST_SIZE));

    PARSER underlyingParser;
    ml::api::CLineifiedXmlInputParser parser(underlyingParser,
                                                  input,
                                                  allDocsSameStructure);

    CVisitor visitor;

    ml::core_t::TTime start(ml::core::CTimeUtils::now());
    LOG_INFO("Starting throughput test at " <<
             ml::core::CTimeUtils::toTimeString(start));

    CPPUNIT_ASSERT(parser.readStream(std::ref(visitor)));

    ml::core_t::TTime end(ml::core::CTimeUtils::now());
    LOG_INFO("Finished throughput test at " <<
             ml::core::CTimeUtils::toTimeString(end));

    CPPUNIT_ASSERT_EQUAL(setupVisitor.recordsPerBlock() * TEST_SIZE, visitor.recordCount());

    LOG_INFO("Parsing " << visitor.recordCount() <<
             " records took " << (end - start) << " seconds");
}

