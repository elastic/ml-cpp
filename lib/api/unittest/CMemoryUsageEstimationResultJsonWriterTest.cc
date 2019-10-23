/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CoreTypes.h>

#include <model/CResourceMonitor.h>

#include <api/CMemoryUsageEstimationResultJsonWriter.h>

#include <rapidjson/document.h>

#include <boost/test/unit_test.hpp>

#include <string>

BOOST_AUTO_TEST_SUITE(CMemoryUsageEstimationResultJsonWriterTest)

using namespace ml;
using namespace api;

BOOST_AUTO_TEST_CASE(testWrite) {
    std::ostringstream sstream;

    // The output writer won't close the JSON structures until is is destroyed
    {
        core::CJsonOutputStreamWrapper wrappedOutStream(sstream);
        CMemoryUsageEstimationResultJsonWriter writer(wrappedOutStream);
        writer.write("16kB", "8kB");
    }

    rapidjson::Document arrayDoc;
    arrayDoc.Parse<rapidjson::kParseDefaultFlags>(sstream.str().c_str());

    BOOST_TEST_REQUIRE(arrayDoc.IsArray());
    BOOST_REQUIRE_EQUAL(rapidjson::SizeType(1), arrayDoc.Size());

    const rapidjson::Value& object = arrayDoc[rapidjson::SizeType(0)];
    BOOST_TEST_REQUIRE(object.IsObject());

    BOOST_TEST_REQUIRE(object.HasMember("expected_memory_without_disk"));
    BOOST_REQUIRE_EQUAL(std::string("16kB"),
                      std::string(object["expected_memory_without_disk"].GetString()));
    BOOST_TEST_REQUIRE(object.HasMember("expected_memory_with_disk"));
    BOOST_REQUIRE_EQUAL(std::string("8kB"),
                      std::string(object["expected_memory_with_disk"].GetString()));
}

BOOST_AUTO_TEST_SUITE_END()
