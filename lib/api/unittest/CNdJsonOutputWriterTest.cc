/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <api/CNdJsonOutputWriter.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CNdJsonOutputWriterTest)

BOOST_AUTO_TEST_CASE(testStringOutput) {
    ml::api::CNdJsonOutputWriter::TStrStrUMap dataRowFields;
    dataRowFields["probability"] = "0.01";
    dataRowFields["normalized_score"] = "2.2";
    ml::api::CNdJsonOutputWriter::TStrStrUMap overrideDataRowFields;
    overrideDataRowFields["normalized_score"] = "3.3";

    ml::api::CNdJsonOutputWriter writer;
    BOOST_TEST_REQUIRE(writer.writeRow(dataRowFields, overrideDataRowFields));

    const std::string& output{writer.internalString()};
    const std::string expected{"{\"probability\":\"0.01\",\"normalized_score\":\"3.3\"}\n"};
    LOG_DEBUG(<< "expected: " << expected);
    LOG_DEBUG(<< "actual  : " << output);

    BOOST_REQUIRE_EQUAL(expected, output);
}

BOOST_AUTO_TEST_CASE(testNumericOutput) {
    ml::api::CNdJsonOutputWriter::TStrStrUMap dataRowFields;
    dataRowFields["probability"] = "0.01";
    dataRowFields["normalized_score"] = "2.2";
    ml::api::CNdJsonOutputWriter::TStrStrUMap overrideDataRowFields;
    overrideDataRowFields["normalized_score"] = "3.3";

    ml::api::CNdJsonOutputWriter writer{{"probability", "normalized_score"}};
    BOOST_TEST_REQUIRE(writer.writeRow(dataRowFields, overrideDataRowFields));

    const std::string& output{writer.internalString()};

    json::value val_ = json::parse(output);
    BOOST_REQUIRE_EQUAL(true, val_.is_object());
    json::object& val = val_.as_object();
    BOOST_REQUIRE_EQUAL(true, val.contains("probability"));
    BOOST_REQUIRE_EQUAL(0.01, val["probability"]);
    BOOST_REQUIRE_EQUAL(true, val.contains("normalized_score"));
    BOOST_REQUIRE_EQUAL(3.3, val["normalized_score"]);
}

BOOST_AUTO_TEST_SUITE_END()
