/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include <api/CNdJsonOutputWriter.h>

#include <boost/test/unit_test.hpp>

#include <sstream>

BOOST_AUTO_TEST_SUITE(CNdJsonOutputWriterTest)

BOOST_AUTO_TEST_CASE(testStringOutput) {
    ml::api::CNdJsonOutputWriter::TStrStrUMap dataRowFields;
    dataRowFields["probability"] = "0.01";
    dataRowFields["normalized_score"] = "2.2";
    ml::api::CNdJsonOutputWriter::TStrStrUMap overrideDataRowFields;
    overrideDataRowFields["normalized_score"] = "3.3";

    ml::api::CNdJsonOutputWriter writer;
    BOOST_TEST_REQUIRE(writer.writeRow(dataRowFields, overrideDataRowFields));

    const std::string& output = writer.internalString();

    BOOST_REQUIRE_EQUAL(
        std::string("{\"probability\":\"0.01\",\"normalized_score\":\"3.3\"}\n"), output);
}

BOOST_AUTO_TEST_CASE(testNumericOutput) {
    ml::api::CNdJsonOutputWriter::TStrStrUMap dataRowFields;
    dataRowFields["probability"] = "0.01";
    dataRowFields["normalized_score"] = "2.2";
    ml::api::CNdJsonOutputWriter::TStrStrUMap overrideDataRowFields;
    overrideDataRowFields["normalized_score"] = "3.3";

    ml::api::CNdJsonOutputWriter writer({"probability", "normalized_score"});
    BOOST_TEST_REQUIRE(writer.writeRow(dataRowFields, overrideDataRowFields));

    const std::string& output = writer.internalString();

    BOOST_REQUIRE_EQUAL(
        std::string("{\"probability\":0.01,\"normalized_score\":3.3}\n"), output);
}

BOOST_AUTO_TEST_SUITE_END()
