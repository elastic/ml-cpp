/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CResultNormalizerTest.h"

#include <core/CLogger.h>

#include <model/CAnomalyDetectorModelConfig.h>

#include <api/CCsvInputParser.h>
#include <api/CLineifiedJsonOutputWriter.h>
#include <api/CResultNormalizer.h>

#include <rapidjson/document.h>

#include <boost/bind.hpp>

#include <fstream>
#include <string>
#include <sstream>

CppUnit::Test *CResultNormalizerTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CResultNormalizerTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CResultNormalizerTest>(
                                   "CResultNormalizerTest::testInitNormalizer",
                                   &CResultNormalizerTest::testInitNormalizer) );

    return suiteOfTests;
}

void CResultNormalizerTest::testInitNormalizer(void)
{
    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(3600);

    ml::api::CLineifiedJsonOutputWriter outputWriter;

    ml::api::CResultNormalizer normalizer(modelConfig, outputWriter);

    CPPUNIT_ASSERT(normalizer.initNormalizer("testfiles/quantilesState.json"));

    std::ifstream inputStrm("testfiles/normalizerInput.csv");
    ml::api::CCsvInputParser inputParser(inputStrm, ml::api::CCsvInputParser::COMMA);
    CPPUNIT_ASSERT(inputParser.readStream(boost::bind(&ml::api::CResultNormalizer::handleRecord,
                                                      &normalizer,
                                                      _1)));

    std::string results(outputWriter.internalString());
    LOG_DEBUG("Results:\n" << results);

    // Results are new line separated so read all the docs into an  array
    std::vector<rapidjson::Document> resultDocs;
    std::stringstream ss(results);
    std::string docString;
    while (std::getline(ss, docString))
    {
        resultDocs.emplace_back();
        resultDocs.back().Parse<rapidjson::kParseDefaultFlags>(docString.c_str());
    }

    CPPUNIT_ASSERT_EQUAL(std::vector<rapidjson::Document>::size_type{38}, resultDocs.size());

    // The maximum bucketTime influencer probability in the Savvis data used to initialise
    // the normaliser is 2.56098e-205, so this should map to the highest normalised
    // score which is 98.28496
    {
        const rapidjson::Document &doc = resultDocs[0];
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("2.56098e-205"), std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("bucketTime"), std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("root"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("98.28496"), std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document &doc = resultDocs[1];
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("2.93761e-203"), std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("status"), std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("inflb"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("97.26764"), std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document &doc = resultDocs[2];
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("5.56572e-204"), std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("status"), std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("infl"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("98.56057"), std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document &doc = resultDocs[4];
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("count"), std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("1e-300"), std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("status"), std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("leaf"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("99.19481"), std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document &doc = resultDocs[15];
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("1e-10"), std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("bucketTime"), std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("root"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("31.20283"), std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document &doc = resultDocs[35];
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("1"), std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("bucketTime"), std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("root"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("0"), std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document &doc = resultDocs[36];
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("1"), std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("status"), std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("infl"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("0"), std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document &doc = resultDocs[37];
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("count"), std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("1"), std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("status"), std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("leaf"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("0"), std::string(doc["normalized_score"].GetString()));
    }
}
