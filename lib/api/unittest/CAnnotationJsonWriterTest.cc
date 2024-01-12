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

#include <core/CJsonOutputStreamWrapper.h>

#include <model/CAnnotation.h>

#include <api/CAnnotationJsonWriter.h>

#include <boost/json.hpp>
#include <boost/test/unit_test.hpp>

#include <sstream>

namespace json = boost::json;

BOOST_AUTO_TEST_SUITE(CAnnotationJsonWriterTest)

BOOST_AUTO_TEST_CASE(testWrite) {
    std::ostringstream sstream;

    {
        ml::core::CJsonOutputStreamWrapper outputStream{sstream};
        ml::api::CAnnotationJsonWriter writer{outputStream};

        ml::model::CAnnotation annotation{1,
                                          ml::model::CAnnotation::E_ModelChange,
                                          "annotation text",
                                          2,
                                          "pName",
                                          "pValue",
                                          "oName",
                                          "oValue",
                                          "bName",
                                          "bValue"};
        writer.writeResult("job-id", annotation);
    }

    LOG_DEBUG(<< "annotation: " << sstream.str());
    json::error_code ec;
    json::value jv = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(jv.is_array());
    json::array doc = jv.as_array();
    const json::value& firstElement_{doc[0]};
    BOOST_TEST_REQUIRE(firstElement_.is_object());
    const json::object& firstElement = firstElement_.as_object();
    BOOST_TEST_REQUIRE(firstElement.contains("annotation"));
    const json::value& annotation_{firstElement.at("annotation")};
    BOOST_TEST_REQUIRE(annotation_.is_object());
    const json::object& annotation = annotation_.as_object();
    BOOST_TEST_REQUIRE(annotation.contains("job_id"));
    BOOST_REQUIRE_EQUAL(std::string{"job-id"}, annotation.at("job_id").as_string());
    BOOST_TEST_REQUIRE(annotation.contains("timestamp"));
    BOOST_REQUIRE_EQUAL(1000, annotation.at("timestamp").as_int64());
    BOOST_TEST_REQUIRE(annotation.contains("detector_index"));
    BOOST_REQUIRE_EQUAL(2, annotation.at("detector_index").as_int64());
    BOOST_TEST_REQUIRE(annotation.contains("partition_field_name"));
    BOOST_REQUIRE_EQUAL(std::string{"pName"},
                        annotation.at("partition_field_name").as_string());
    BOOST_TEST_REQUIRE(annotation.contains("partition_field_value"));
    BOOST_REQUIRE_EQUAL(std::string{"pValue"},
                        annotation.at("partition_field_value").as_string());
    BOOST_TEST_REQUIRE(annotation.contains("over_field_name"));
    BOOST_REQUIRE_EQUAL(std::string{"oName"}, annotation.at("over_field_name").as_string());
    BOOST_TEST_REQUIRE(annotation.contains("over_field_value"));
    BOOST_REQUIRE_EQUAL(std::string{"oValue"}, annotation.at("over_field_value").as_string());
    BOOST_TEST_REQUIRE(annotation.contains("by_field_name"));
    BOOST_REQUIRE_EQUAL(std::string{"bName"}, annotation.at("by_field_name").as_string());
    BOOST_TEST_REQUIRE(annotation.contains("by_field_value"));
    BOOST_REQUIRE_EQUAL(std::string{"bValue"}, annotation.at("by_field_value").as_string());
    BOOST_TEST_REQUIRE(annotation.contains("annotation"));
    BOOST_REQUIRE_EQUAL(std::string{"annotation text"},
                        annotation.at("annotation").as_string());
    BOOST_TEST_REQUIRE(annotation.contains("event"));
    BOOST_REQUIRE_EQUAL(std::string{"model_change"}, annotation.at("event").as_string());
}

BOOST_AUTO_TEST_SUITE_END()
