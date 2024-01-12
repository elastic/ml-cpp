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

#include <model/CModelPlotData.h>
#include <model/ModelTypes.h>

#include <api/CModelPlotDataJsonWriter.h>

#include <test/BoostTestCloseAbsolute.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CModelPlotDataJsonWriterTest)

BOOST_AUTO_TEST_CASE(testWriteFlat) {
    std::ostringstream sstream;

    {
        ml::core::CJsonOutputStreamWrapper outputStream(sstream);
        ml::api::CModelPlotDataJsonWriter writer(outputStream);

        ml::model::CModelPlotData plotData(1, "pName", "pValue", "", "bName", 300, 1);
        plotData.get(ml::model_t::E_IndividualCountByBucketAndPerson, "bName") =
            ml::model::CModelPlotData::SByFieldData(1.0, 2.0, 3.0);

        writer.writeFlat("job-id", plotData);
    }

    json::error_code ec;
    json::value doc = json::parse(sstream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(doc.is_array());
    
    const json::value& firstElement_ = doc.as_array()[0];
    const json::object& firstElement = firstElement_.as_object();
    BOOST_TEST_REQUIRE(firstElement.contains("model_plot"));
    const json::value& modelPlot_ = firstElement.at("model_plot");
    const json::object& modelPlot = modelPlot_.as_object();
    BOOST_TEST_REQUIRE(modelPlot.contains("job_id"));
    BOOST_REQUIRE_EQUAL(std::string("job-id"),
                        std::string(modelPlot.at("job_id").as_string()));
    BOOST_TEST_REQUIRE(modelPlot.contains("model_feature"));
    BOOST_REQUIRE_EQUAL(std::string("'count per bucket by person'"),
                        std::string(modelPlot.at("model_feature").as_string()));
    BOOST_TEST_REQUIRE(modelPlot.contains("timestamp"));
    BOOST_REQUIRE_EQUAL(1000, modelPlot.at("timestamp").as_uint64());
    BOOST_TEST_REQUIRE(modelPlot.contains("partition_field_name"));
    BOOST_REQUIRE_EQUAL(std::string("pName"),
                        std::string(modelPlot.at("partition_field_name").as_string()));
    BOOST_TEST_REQUIRE(modelPlot.contains("partition_field_value"));
    BOOST_REQUIRE_EQUAL(std::string("pValue"),
                        std::string(modelPlot.at("partition_field_value").as_string()));
    BOOST_TEST_REQUIRE(modelPlot.contains("by_field_name"));
    BOOST_REQUIRE_EQUAL(std::string("bName"),
                        std::string(modelPlot.at("by_field_name").as_string()));
    BOOST_TEST_REQUIRE(modelPlot.contains("by_field_value"));
    BOOST_REQUIRE_EQUAL(std::string("bName"),
                        std::string(modelPlot.at("by_field_value").as_string()));
    BOOST_TEST_REQUIRE(modelPlot.contains("model_lower"));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, modelPlot.at("model_lower").as_double(), 0.01);
    BOOST_TEST_REQUIRE(modelPlot.contains("model_upper"));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(2.0, modelPlot.at("model_upper").as_double(), 0.01);
    BOOST_TEST_REQUIRE(modelPlot.contains("model_median"));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(3.0, modelPlot.at("model_median").as_double(), 0.01);
    BOOST_TEST_REQUIRE(modelPlot.contains("bucket_span"));
    BOOST_REQUIRE_EQUAL(300, modelPlot.at("bucket_span").as_uint64());
}

BOOST_AUTO_TEST_SUITE_END()
