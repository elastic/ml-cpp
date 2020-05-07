/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CJsonOutputStreamWrapper.h>

#include <model/CAnnotation.h>
#include <model/ModelTypes.h>

#include <api/CAnnotationJsonWriter.h>

#include <test/BoostTestCloseAbsolute.h>

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CAnnotationJsonWriterTest)

BOOST_AUTO_TEST_CASE(testWrite) {
    std::ostringstream sstream;

    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CAnnotationJsonWriter writer(outputStream);

        ml::model::CAnnotation annotation(1, "annotation text", 2, "pName", "pValue",
                                          "oName", "oValue", "bName", "bValue");
        writer.writeResult("job-id", annotation);
    }

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(sstream.str());
    BOOST_TEST_REQUIRE(!doc.HasParseError());
    const rapidjson::Value& firstElement = doc[0];
    BOOST_TEST_REQUIRE(firstElement.HasMember("annotation"));
    const rapidjson::Value& annotation = firstElement["annotation"];
    BOOST_TEST_REQUIRE(annotation.HasMember("job_id"));
    BOOST_REQUIRE_EQUAL(std::string("job-id"),
                        std::string(annotation["job_id"].GetString()));
    BOOST_TEST_REQUIRE(annotation.HasMember("timestamp"));
    BOOST_REQUIRE_EQUAL(int64_t(1000), annotation["timestamp"].GetInt64());
    BOOST_TEST_REQUIRE(annotation.HasMember("detector_index"));
    BOOST_REQUIRE_EQUAL(2, annotation["detector_index"].GetInt());
    BOOST_TEST_REQUIRE(annotation.HasMember("partition_field_name"));
    BOOST_REQUIRE_EQUAL(std::string("pName"),
                        std::string(annotation["partition_field_name"].GetString()));
    BOOST_TEST_REQUIRE(annotation.HasMember("partition_field_value"));
    BOOST_REQUIRE_EQUAL(std::string("pValue"),
                        std::string(annotation["partition_field_value"].GetString()));
    BOOST_TEST_REQUIRE(annotation.HasMember("over_field_name"));
    BOOST_REQUIRE_EQUAL(std::string("oName"),
                        std::string(annotation["over_field_name"].GetString()));
    BOOST_TEST_REQUIRE(annotation.HasMember("over_field_value"));
    BOOST_REQUIRE_EQUAL(std::string("oValue"),
                        std::string(annotation["over_field_value"].GetString()));
    BOOST_TEST_REQUIRE(annotation.HasMember("by_field_name"));
    BOOST_REQUIRE_EQUAL(std::string("bName"),
                        std::string(annotation["by_field_name"].GetString()));
    BOOST_TEST_REQUIRE(annotation.HasMember("by_field_value"));
    BOOST_REQUIRE_EQUAL(std::string("bValue"),
                        std::string(annotation["by_field_value"].GetString()));
    BOOST_TEST_REQUIRE(annotation.HasMember("annotation"));
    BOOST_REQUIRE_EQUAL(std::string("annotation text"),
                        std::string(annotation["annotation"].GetString()));
}

BOOST_AUTO_TEST_SUITE_END()
