/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CModelPlotDataJsonWriterTest.h"

#include <core/CJsonOutputStreamWrapper.h>

#include <api/CModelPlotDataJsonWriter.h>

#include <model/CModelPlotData.h>
#include <model/ModelTypes.h>

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>

CppUnit::Test* CModelPlotDataJsonWriterTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CModelPlotDataJsonWriterTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CModelPlotDataJsonWriterTest>(
                                   "CModelPlotDataJsonWriterTest::testWriteFlat",
                                   &CModelPlotDataJsonWriterTest::testWriteFlat) );

    return suiteOfTests;
}

void CModelPlotDataJsonWriterTest::testWriteFlat()
{
    std::ostringstream sstream;

    {
        ml::core::CJsonOutputStreamWrapper outputStream (sstream);
        ml::api::CModelPlotDataJsonWriter writer(outputStream);

        ml::model::CModelPlotData plotData(1, "pName", "pValue", "", "bName", 300, 1);
        plotData.get(ml::model_t::E_IndividualCountByBucketAndPerson, "bName") = ml::model::CModelPlotData::SByFieldData(1.0,  2.0, 3.0);

        writer.writeFlat("job-id", plotData);
    }

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(sstream.str());
    CPPUNIT_ASSERT(!doc.HasParseError());
    const rapidjson::Value &firstElement = doc[0];
    CPPUNIT_ASSERT(firstElement.HasMember("model_plot"));
    const rapidjson::Value &modelPlot = firstElement["model_plot"];
    CPPUNIT_ASSERT(modelPlot.HasMember("job_id"));
    CPPUNIT_ASSERT_EQUAL(std::string("job-id"), std::string(modelPlot["job_id"].GetString()));
    CPPUNIT_ASSERT(modelPlot.HasMember("model_feature"));
    CPPUNIT_ASSERT_EQUAL(std::string("'count per bucket by person'"),
        std::string(modelPlot["model_feature"].GetString()));
    CPPUNIT_ASSERT(modelPlot.HasMember("timestamp"));
    CPPUNIT_ASSERT_EQUAL(int64_t(1000), modelPlot["timestamp"].GetInt64());
    CPPUNIT_ASSERT(modelPlot.HasMember("partition_field_name"));
    CPPUNIT_ASSERT_EQUAL(std::string("pName"), std::string(modelPlot["partition_field_name"].GetString()));
    CPPUNIT_ASSERT(modelPlot.HasMember("partition_field_value"));
    CPPUNIT_ASSERT_EQUAL(std::string("pValue"), std::string(modelPlot["partition_field_value"].GetString()));
    CPPUNIT_ASSERT(modelPlot.HasMember("by_field_name"));
    CPPUNIT_ASSERT_EQUAL(std::string("bName"), std::string(modelPlot["by_field_name"].GetString()));
    CPPUNIT_ASSERT(modelPlot.HasMember("by_field_value"));
    CPPUNIT_ASSERT_EQUAL(std::string("bName"), std::string(modelPlot["by_field_value"].GetString()));
    CPPUNIT_ASSERT(modelPlot.HasMember("model_lower"));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, modelPlot["model_lower"].GetDouble(), 0.01);
    CPPUNIT_ASSERT(modelPlot.HasMember("model_upper"));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, modelPlot["model_upper"].GetDouble(), 0.01);
    CPPUNIT_ASSERT(modelPlot.HasMember("model_median"));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, modelPlot["model_median"].GetDouble(), 0.01);
    CPPUNIT_ASSERT(modelPlot.HasMember("bucket_span"));
    CPPUNIT_ASSERT_EQUAL(int64_t(300), modelPlot["bucket_span"].GetInt64());
}
