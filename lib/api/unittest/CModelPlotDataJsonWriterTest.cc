/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CJsonOutputStreamWrapper.h>

#include <model/CModelPlotData.h>
#include <model/ModelTypes.h>

#include <api/CModelPlotDataJsonWriter.h>

#include <test/BoostTestCloseAbsolute.h>

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>

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

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(sstream.str());
    BOOST_TEST(!doc.HasParseError());
    const rapidjson::Value& firstElement = doc[0];
    BOOST_TEST(firstElement.HasMember("model_plot"));
    const rapidjson::Value& modelPlot = firstElement["model_plot"];
    BOOST_TEST(modelPlot.HasMember("job_id"));
    BOOST_CHECK_EQUAL(std::string("job-id"),
                         std::string(modelPlot["job_id"].GetString()));
    BOOST_TEST(modelPlot.HasMember("model_feature"));
    BOOST_CHECK_EQUAL(std::string("'count per bucket by person'"),
                         std::string(modelPlot["model_feature"].GetString()));
    BOOST_TEST(modelPlot.HasMember("timestamp"));
    BOOST_CHECK_EQUAL(int64_t(1000), modelPlot["timestamp"].GetInt64());
    BOOST_TEST(modelPlot.HasMember("partition_field_name"));
    BOOST_CHECK_EQUAL(std::string("pName"),
                         std::string(modelPlot["partition_field_name"].GetString()));
    BOOST_TEST(modelPlot.HasMember("partition_field_value"));
    BOOST_CHECK_EQUAL(std::string("pValue"),
                         std::string(modelPlot["partition_field_value"].GetString()));
    BOOST_TEST(modelPlot.HasMember("by_field_name"));
    BOOST_CHECK_EQUAL(std::string("bName"),
                         std::string(modelPlot["by_field_name"].GetString()));
    BOOST_TEST(modelPlot.HasMember("by_field_value"));
    BOOST_CHECK_EQUAL(std::string("bName"),
                         std::string(modelPlot["by_field_value"].GetString()));
    BOOST_TEST(modelPlot.HasMember("model_lower"));
    BOOST_CHECK_CLOSE_ABSOLUTE(1.0, modelPlot["model_lower"].GetDouble(), 0.01);
    BOOST_TEST(modelPlot.HasMember("model_upper"));
    BOOST_CHECK_CLOSE_ABSOLUTE(2.0, modelPlot["model_upper"].GetDouble(), 0.01);
    BOOST_TEST(modelPlot.HasMember("model_median"));
    BOOST_CHECK_CLOSE_ABSOLUTE(3.0, modelPlot["model_median"].GetDouble(), 0.01);
    BOOST_TEST(modelPlot.HasMember("bucket_span"));
    BOOST_CHECK_EQUAL(int64_t(300), modelPlot["bucket_span"].GetInt64());
}

BOOST_AUTO_TEST_SUITE_END()
