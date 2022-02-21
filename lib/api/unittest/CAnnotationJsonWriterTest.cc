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

#include <rapidjson/document.h>

#include <boost/test/unit_test.hpp>

#include <sstream>

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

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(sstream.str());
    BOOST_TEST_REQUIRE(doc.HasParseError() == false);
    const rapidjson::Value& firstElement{doc[0]};
    BOOST_TEST_REQUIRE(firstElement.HasMember("annotation"));
    const rapidjson::Value& annotation{firstElement["annotation"]};
    BOOST_TEST_REQUIRE(annotation.HasMember("job_id"));
    BOOST_REQUIRE_EQUAL(std::string{"job-id"}, annotation["job_id"].GetString());
    BOOST_TEST_REQUIRE(annotation.HasMember("timestamp"));
    BOOST_REQUIRE_EQUAL(1000, annotation["timestamp"].GetInt64());
    BOOST_TEST_REQUIRE(annotation.HasMember("detector_index"));
    BOOST_REQUIRE_EQUAL(2, annotation["detector_index"].GetInt());
    BOOST_TEST_REQUIRE(annotation.HasMember("partition_field_name"));
    BOOST_REQUIRE_EQUAL(std::string{"pName"},
                        annotation["partition_field_name"].GetString());
    BOOST_TEST_REQUIRE(annotation.HasMember("partition_field_value"));
    BOOST_REQUIRE_EQUAL(std::string{"pValue"},
                        annotation["partition_field_value"].GetString());
    BOOST_TEST_REQUIRE(annotation.HasMember("over_field_name"));
    BOOST_REQUIRE_EQUAL(std::string{"oName"}, annotation["over_field_name"].GetString());
    BOOST_TEST_REQUIRE(annotation.HasMember("over_field_value"));
    BOOST_REQUIRE_EQUAL(std::string{"oValue"}, annotation["over_field_value"].GetString());
    BOOST_TEST_REQUIRE(annotation.HasMember("by_field_name"));
    BOOST_REQUIRE_EQUAL(std::string{"bName"}, annotation["by_field_name"].GetString());
    BOOST_TEST_REQUIRE(annotation.HasMember("by_field_value"));
    BOOST_REQUIRE_EQUAL(std::string{"bValue"}, annotation["by_field_value"].GetString());
    BOOST_TEST_REQUIRE(annotation.HasMember("annotation"));
    BOOST_REQUIRE_EQUAL(std::string{"annotation text"},
                        annotation["annotation"].GetString());
    BOOST_TEST_REQUIRE(annotation.HasMember("event"));
    BOOST_REQUIRE_EQUAL(std::string{"model_change"}, annotation["event"].GetString());
}

BOOST_AUTO_TEST_SUITE_END()
