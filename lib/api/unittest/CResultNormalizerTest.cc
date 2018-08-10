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
#include <sstream>
#include <string>

CppUnit::Test* CResultNormalizerTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CResultNormalizerTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CResultNormalizerTest>(
        "CResultNormalizerTest::testInitNormalizerPartitioned",
        &CResultNormalizerTest::testInitNormalizerPartitioned));

    suiteOfTests->addTest(new CppUnit::TestCaller<CResultNormalizerTest>(
        "CResultNormalizerTest::testInitNormalizer", &CResultNormalizerTest::testInitNormalizer));

    return suiteOfTests;
}

void CResultNormalizerTest::testInitNormalizerPartitioned() {
    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(900);

    ml::api::CLineifiedJsonOutputWriter outputWriter;

    ml::api::CResultNormalizer normalizer(modelConfig, outputWriter);

    CPPUNIT_ASSERT(normalizer.initNormalizer("testfiles/new_quantilesState.json"));

    std::ifstream inputStrm("testfiles/new_normalizerInput.csv");
    ml::api::CCsvInputParser inputParser(inputStrm, ml::api::CCsvInputParser::COMMA);
    CPPUNIT_ASSERT(inputParser.readStream(
        boost::bind(&ml::api::CResultNormalizer::handleRecord, &normalizer, _1)));

    std::string results(outputWriter.internalString());
    LOG_INFO(<< "Results:\n" << results);

    // Results are new line separated so read all the docs into an  array
    std::vector<rapidjson::Document> resultDocs;
    std::stringstream ss(results);
    std::string docString;
    while (std::getline(ss, docString)) {
        resultDocs.emplace_back();
        resultDocs.back().Parse<rapidjson::kParseDefaultFlags>(docString.c_str());
    }

    CPPUNIT_ASSERT_EQUAL(std::vector<rapidjson::Document>::size_type{327},
                         resultDocs.size());

    // The maximum bucket_time influencer probability in the farequote data used to initialise
    // the normaliser is 2.1305076675260463E-24, so this should map to the highest normalised
    // score which is 90.69091
    {
        const rapidjson::Document& doc = resultDocs[13];
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("2.1305076675260463E-24"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("bucket_time"),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("root"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("90.69091"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[16];
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("1.90875417733942E-22"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("airline"),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("inflb"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("69.60219"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[18];
        CPPUNIT_ASSERT_EQUAL(std::string("responsetime"),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("high_mean"),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("0.003824460431046938"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("airline"),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("ASA"),
                             std::string(doc["partition_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("leaf"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("44.29573"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[19];
        CPPUNIT_ASSERT_EQUAL(std::string("responsetime"),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("high_mean"),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("0.035787369763616045"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("airline"),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("FFT"),
                             std::string(doc["partition_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("leaf"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("0"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[20];
        CPPUNIT_ASSERT_EQUAL(std::string("responsetime"),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("high_mean"),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("0.013241693005461058"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("airline"),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("SWA"),
                             std::string(doc["partition_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("leaf"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("25.31053"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[41];
        CPPUNIT_ASSERT_EQUAL(std::string("responsetime"),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("high_mean"),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("4.026858488865555E-4"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("airline"),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("TRS"),
                             std::string(doc["partition_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("leaf"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("78.70602"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[61];
        CPPUNIT_ASSERT_EQUAL(std::string("responsetime"),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("high_mean"),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("0.0014127543730254476"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("airline"),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("UAL"),
                             std::string(doc["partition_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("leaf"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("59.51925"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[65];
        CPPUNIT_ASSERT_EQUAL(std::string("responsetime"),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("high_mean"),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("4.440295404321955E-4"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("airline"),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("JAL"),
                             std::string(doc["partition_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("leaf"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("77.212"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[74];
        CPPUNIT_ASSERT_EQUAL(std::string("responsetime"),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("high_mean"),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("0.011347202666523165"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("airline"),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("EGF"),
                             std::string(doc["partition_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("leaf"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("27.67076"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[84];
        CPPUNIT_ASSERT_EQUAL(std::string("responsetime"),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("high_mean"),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("7.891442535915233E-31"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("airline"),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("AAL"),
                             std::string(doc["partition_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("leaf"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("99.22667"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[138];
        CPPUNIT_ASSERT_EQUAL(std::string("responsetime"),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("high_mean"),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("0.0300716991931514"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("airline"),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("EGF"),
                             std::string(doc["partition_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("leaf"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("8.178799"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[166];
        CPPUNIT_ASSERT_EQUAL(std::string("responsetime"),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("high_mean"),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("0.008751921671499477"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["person_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("airline"),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("KLM"),
                             std::string(doc["partition_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("leaf"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("25.34194"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[227];
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("5.538073913329369E-4"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("airline"),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("JAL"),
                             std::string(doc["person_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("infl"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("73.8348"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[301];
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("0.009557853111806711"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("airline"),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("EGF"),
                             std::string(doc["person_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("infl"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("30.29406"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[306];
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("1.8102856956640726E-23"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("airline"),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("AAL"),
                             std::string(doc["person_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("infl"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("96.89889"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[317];
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("0.0021402294772877688"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("airline"),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("KLM"),
                             std::string(doc["person_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_value"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("infl"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("53.16964"),
                             std::string(doc["normalized_score"].GetString()));
    }
}

void CResultNormalizerTest::testInitNormalizer() {
    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(3600);

    ml::api::CLineifiedJsonOutputWriter outputWriter;

    ml::api::CResultNormalizer normalizer(modelConfig, outputWriter);

    CPPUNIT_ASSERT(normalizer.initNormalizer("testfiles/quantilesState.json"));

    std::ifstream inputStrm("testfiles/normalizerInput.csv");
    ml::api::CCsvInputParser inputParser(inputStrm, ml::api::CCsvInputParser::COMMA);
    CPPUNIT_ASSERT(inputParser.readStream(
        boost::bind(&ml::api::CResultNormalizer::handleRecord, &normalizer, _1)));

    std::string results(outputWriter.internalString());
    LOG_DEBUG(<< "Results:\n" << results);

    // Results are new line separated so read all the docs into an  array
    std::vector<rapidjson::Document> resultDocs;
    std::stringstream ss(results);
    std::string docString;
    while (std::getline(ss, docString)) {
        resultDocs.emplace_back();
        resultDocs.back().Parse<rapidjson::kParseDefaultFlags>(docString.c_str());
    }

    CPPUNIT_ASSERT_EQUAL(std::vector<rapidjson::Document>::size_type{38},
                         resultDocs.size());

    // The maximum bucketTime influencer probability in the Savvis data used to initialise
    // the normaliser is 2.56098e-205, so this should map to the highest normalised
    // score which is 98.28496
    {
        const rapidjson::Document& doc = resultDocs[0];
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("2.56098e-205"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("bucketTime"),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("root"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("98.28496"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[1];
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("2.93761e-203"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("status"),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("inflb"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("97.26764"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[2];
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("5.56572e-204"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("status"),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("infl"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("98.56057"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[4];
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("count"),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("1e-300"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("status"),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("leaf"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("99.19481"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[15];
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("1e-10"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("bucketTime"),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("root"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("31.20283"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[35];
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("1"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("bucketTime"),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("root"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("0"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[36];
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("1"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("status"),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("infl"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("0"),
                             std::string(doc["normalized_score"].GetString()));
    }
    {
        const rapidjson::Document& doc = resultDocs[37];
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["value_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("count"),
                             std::string(doc["function_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("1"),
                             std::string(doc["probability"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("status"),
                             std::string(doc["person_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string(""),
                             std::string(doc["partition_field_name"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("leaf"), std::string(doc["level"].GetString()));
        CPPUNIT_ASSERT_EQUAL(std::string("0"),
                             std::string(doc["normalized_score"].GetString()));
    }
}
