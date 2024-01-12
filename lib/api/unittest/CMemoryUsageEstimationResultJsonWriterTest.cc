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
#include <core/CoreTypes.h>

#include <model/CResourceMonitor.h>

#include <api/CMemoryUsageEstimationResultJsonWriter.h>

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
        writer.write("16mb", "8mb");
    }

    json::error_code ec;
    json::value arrayDoc = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(arrayDoc.is_array());

    BOOST_REQUIRE_EQUAL(std::size_t(1), arrayDoc.as_array().size());

    const json::value& object_ = arrayDoc.as_array()[std::size_t(0)];
    BOOST_TEST_REQUIRE(object_.is_object());
    const json::object& object = object_.as_object();
    BOOST_TEST_REQUIRE(object.contains("expected_memory_without_disk"));
    BOOST_REQUIRE_EQUAL(std::string("16mb"),
                        std::string(object.at("expected_memory_without_disk").as_string()));
    BOOST_TEST_REQUIRE(object.contains("expected_memory_with_disk"));
    BOOST_REQUIRE_EQUAL(std::string("8mb"),
                        std::string(object.at("expected_memory_with_disk").as_string()));
}

BOOST_AUTO_TEST_SUITE_END()
