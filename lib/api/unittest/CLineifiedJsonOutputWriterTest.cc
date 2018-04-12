/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CLineifiedJsonOutputWriterTest.h"

#include <core/CLogger.h>

#include <api/CLineifiedJsonOutputWriter.h>

#include <sstream>

CppUnit::Test* CLineifiedJsonOutputWriterTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CLineifiedJsonOutputWriterTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CLineifiedJsonOutputWriterTest>(
        "CLineifiedJsonOutputWriterTest::testStringOutput",
        &CLineifiedJsonOutputWriterTest::testStringOutput));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLineifiedJsonOutputWriterTest>(
        "CLineifiedJsonOutputWriterTest::testNumericOutput",
        &CLineifiedJsonOutputWriterTest::testNumericOutput));

    return suiteOfTests;
}

void CLineifiedJsonOutputWriterTest::testStringOutput() {
    ml::api::CLineifiedJsonOutputWriter::TStrStrUMap dataRowFields;
    dataRowFields["probability"] = "0.01";
    dataRowFields["normalized_score"] = "2.2";
    ml::api::CLineifiedJsonOutputWriter::TStrStrUMap overrideDataRowFields;
    overrideDataRowFields["normalized_score"] = "3.3";

    ml::api::CLineifiedJsonOutputWriter writer;
    CPPUNIT_ASSERT(writer.writeRow(dataRowFields, overrideDataRowFields));

    const std::string& output = writer.internalString();

    CPPUNIT_ASSERT_EQUAL(std::string("{\"probability\":\"0.01\",\"normalized_"
                                     "score\":\"3.3\"}\n"),
                         output);
}

void CLineifiedJsonOutputWriterTest::testNumericOutput() {
    ml::api::CLineifiedJsonOutputWriter::TStrStrUMap dataRowFields;
    dataRowFields["probability"] = "0.01";
    dataRowFields["normalized_score"] = "2.2";
    ml::api::CLineifiedJsonOutputWriter::TStrStrUMap overrideDataRowFields;
    overrideDataRowFields["normalized_score"] = "3.3";

    ml::api::CLineifiedJsonOutputWriter writer({"probability", "normalized_score"});
    CPPUNIT_ASSERT(writer.writeRow(dataRowFields, overrideDataRowFields));

    const std::string& output = writer.internalString();

    CPPUNIT_ASSERT_EQUAL(
        std::string("{\"probability\":0.01,\"normalized_score\":3.3}\n"), output);
}
