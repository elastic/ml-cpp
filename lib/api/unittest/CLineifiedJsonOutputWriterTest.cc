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
#include "CLineifiedJsonOutputWriterTest.h"

#include <core/CLogger.h>

#include <api/CLineifiedJsonOutputWriter.h>

#include <sstream>


CppUnit::Test *CLineifiedJsonOutputWriterTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CLineifiedJsonOutputWriterTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CLineifiedJsonOutputWriterTest>(
                                   "CLineifiedJsonOutputWriterTest::testStringOutput",
                                   &CLineifiedJsonOutputWriterTest::testStringOutput) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CLineifiedJsonOutputWriterTest>(
                                   "CLineifiedJsonOutputWriterTest::testNumericOutput",
                                   &CLineifiedJsonOutputWriterTest::testNumericOutput) );

    return suiteOfTests;
}

void CLineifiedJsonOutputWriterTest::testStringOutput(void)
{
    ml::api::CLineifiedJsonOutputWriter::TStrStrUMap dataRowFields;
    dataRowFields["probability"] = "0.01";
    dataRowFields["normalized_score"] = "2.2";
    ml::api::CLineifiedJsonOutputWriter::TStrStrUMap overrideDataRowFields;
    overrideDataRowFields["normalized_score"] = "3.3";

    ml::api::CLineifiedJsonOutputWriter writer;
    CPPUNIT_ASSERT(writer.writeRow(dataRowFields, overrideDataRowFields));

    const std::string &output = writer.internalString();

    CPPUNIT_ASSERT_EQUAL(std::string("{\"probability\":\"0.01\",\"normalized_score\":\"3.3\"}\n"), output);
}

void CLineifiedJsonOutputWriterTest::testNumericOutput(void)
{
    ml::api::CLineifiedJsonOutputWriter::TStrStrUMap dataRowFields;
    dataRowFields["probability"] = "0.01";
    dataRowFields["normalized_score"] = "2.2";
    ml::api::CLineifiedJsonOutputWriter::TStrStrUMap overrideDataRowFields;
    overrideDataRowFields["normalized_score"] = "3.3";

    ml::api::CLineifiedJsonOutputWriter writer({ "probability", "normalized_score" });
    CPPUNIT_ASSERT(writer.writeRow(dataRowFields, overrideDataRowFields));

    const std::string &output = writer.internalString();

    CPPUNIT_ASSERT_EQUAL(std::string("{\"probability\":0.01,\"normalized_score\":3.3}\n"), output);
}

