/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CNdJsonOutputWriterTest.h"

#include <core/CLogger.h>

#include <api/CNdJsonOutputWriter.h>

#include <sstream>

CppUnit::Test* CNdJsonOutputWriterTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CNdJsonOutputWriterTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CNdJsonOutputWriterTest>(
        "CNdJsonOutputWriterTest::testStringOutput", &CNdJsonOutputWriterTest::testStringOutput));
    suiteOfTests->addTest(new CppUnit::TestCaller<CNdJsonOutputWriterTest>(
        "CNdJsonOutputWriterTest::testNumericOutput", &CNdJsonOutputWriterTest::testNumericOutput));

    return suiteOfTests;
}

void CNdJsonOutputWriterTest::testStringOutput() {
    ml::api::CNdJsonOutputWriter::TStrStrUMap dataRowFields;
    dataRowFields["probability"] = "0.01";
    dataRowFields["normalized_score"] = "2.2";
    ml::api::CNdJsonOutputWriter::TStrStrUMap overrideDataRowFields;
    overrideDataRowFields["normalized_score"] = "3.3";

    ml::api::CNdJsonOutputWriter writer;
    CPPUNIT_ASSERT(writer.writeRow(dataRowFields, overrideDataRowFields));

    const std::string& output = writer.internalString();

    CPPUNIT_ASSERT_EQUAL(
        std::string("{\"probability\":\"0.01\",\"normalized_score\":\"3.3\"}\n"), output);
}

void CNdJsonOutputWriterTest::testNumericOutput() {
    ml::api::CNdJsonOutputWriter::TStrStrUMap dataRowFields;
    dataRowFields["probability"] = "0.01";
    dataRowFields["normalized_score"] = "2.2";
    ml::api::CNdJsonOutputWriter::TStrStrUMap overrideDataRowFields;
    overrideDataRowFields["normalized_score"] = "3.3";

    ml::api::CNdJsonOutputWriter writer({"probability", "normalized_score"});
    CPPUNIT_ASSERT(writer.writeRow(dataRowFields, overrideDataRowFields));

    const std::string& output = writer.internalString();

    CPPUNIT_ASSERT_EQUAL(
        std::string("{\"probability\":0.01,\"normalized_score\":3.3}\n"), output);
}
