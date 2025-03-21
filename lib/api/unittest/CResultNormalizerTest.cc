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

#include <core/CLogger.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <api/CCsvInputParser.h>
#include <api/CNdJsonOutputWriter.h>
#include <api/CResultNormalizer.h>

#include <boost/test/unit_test.hpp>

#include <fstream>
#include <sstream>
#include <string>

BOOST_AUTO_TEST_SUITE(CResultNormalizerTest)

BOOST_AUTO_TEST_CASE(testInitNormalizerPartitioned) {
    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(900);

    ml::api::CNdJsonOutputWriter outputWriter;

    ml::model::CLimits limits(false);
    ml::api::CResultNormalizer normalizer(modelConfig, outputWriter, limits);

    BOOST_TEST_REQUIRE(normalizer.initNormalizer("testfiles/new_quantilesState.json"));
    LOG_DEBUG(<< "normalizer initialized");

    std::ifstream inputStrm("testfiles/new_normalizerInput.csv");
    ml::api::CCsvInputParser inputParser(inputStrm);
    BOOST_TEST_REQUIRE(inputParser.readStreamIntoMaps(std::bind(
        &ml::api::CResultNormalizer::handleRecord, &normalizer, std::placeholders::_1)));

    std::string results(outputWriter.internalString());
    LOG_INFO(<< "Results:\n" << results);

    // Results are new line separated so read all the docs into an array.
    std::vector<json::value> resultDocs;
    std::stringstream ss(results);
    std::string docString;
    json::error_code ec;
    while (std::getline(ss, docString)) {
        json::value doc = json::parse(docString.c_str(), ec);
        BOOST_TEST_REQUIRE(ec.failed() == false);
        resultDocs.push_back(doc);
    }

    BOOST_REQUIRE_EQUAL(std::vector<json::value>::size_type{327}, resultDocs.size());

    // The maximum bucket_time influencer probability in the farequote data used to initialise
    // the normaliser is 2.1305076675260463E-24, so this should map to the highest normalised
    // score which is 93.9542.
    {
        const json::value& doc_ = resultDocs[13];
        const json::object& doc = doc_.as_object();
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("2.1305076675260463E-24"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("bucket_time"),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("root"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("93.9542"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[16];
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("1.90875417733942E-22"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("airline"),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("inflb"),
                            std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("81.65058"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[18];
        BOOST_REQUIRE_EQUAL(std::string("responsetime"),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("high_mean"),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("0.003824460431046938"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("airline"),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("ASA"),
                            std::string(doc.at("partition_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("leaf"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("44.29573"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[19];
        BOOST_REQUIRE_EQUAL(std::string("responsetime"),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("high_mean"),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("0.035787369763616045"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("airline"),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("FFT"),
                            std::string(doc.at("partition_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("leaf"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("0"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[20];
        BOOST_REQUIRE_EQUAL(std::string("responsetime"),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("high_mean"),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("0.013241693005461058"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("airline"),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("SWA"),
                            std::string(doc.at("partition_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("leaf"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("25.31053"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[41];
        BOOST_REQUIRE_EQUAL(std::string("responsetime"),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("high_mean"),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("4.026858488865555E-4"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("airline"),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("TRS"),
                            std::string(doc.at("partition_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("leaf"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("78.70602"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[61];
        BOOST_REQUIRE_EQUAL(std::string("responsetime"),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("high_mean"),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("0.0014127543730254476"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("airline"),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("UAL"),
                            std::string(doc.at("partition_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("leaf"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("59.51925"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[65];
        BOOST_REQUIRE_EQUAL(std::string("responsetime"),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("high_mean"),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("4.440295404321955E-4"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("airline"),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("JAL"),
                            std::string(doc.at("partition_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("leaf"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("77.212"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[74];
        BOOST_REQUIRE_EQUAL(std::string("responsetime"),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("high_mean"),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("0.011347202666523165"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("airline"),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("EGF"),
                            std::string(doc.at("partition_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("leaf"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("27.67076"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[84];
        BOOST_REQUIRE_EQUAL(std::string("responsetime"),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("high_mean"),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("7.891442535915233E-31"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("airline"),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("AAL"),
                            std::string(doc.at("partition_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("leaf"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("99.61332"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[138];
        BOOST_REQUIRE_EQUAL(std::string("responsetime"),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("high_mean"),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("0.0300716991931514"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("airline"),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("EGF"),
                            std::string(doc.at("partition_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("leaf"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("8.178497"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[166];
        BOOST_REQUIRE_EQUAL(std::string("responsetime"),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("high_mean"),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("0.008751921671499477"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("person_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("airline"),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("KLM"),
                            std::string(doc.at("partition_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("leaf"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("25.34066"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[227];
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("5.538073913329369E-4"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("airline"),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("JAL"),
                            std::string(doc.at("person_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("infl"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("73.8348"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[301];
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("0.009557853111806711"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("airline"),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("EGF"),
                            std::string(doc.at("person_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("infl"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("30.29406"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[306];
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("1.8102856956640726E-23"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("airline"),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("AAL"),
                            std::string(doc.at("person_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("infl"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("98.44925"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[317];
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("0.0021402294772877688"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("airline"),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("KLM"),
                            std::string(doc.at("person_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_value").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("infl"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("53.16964"),
                            std::string(doc.at("normalized_score").as_string()));
    }
}

BOOST_AUTO_TEST_CASE(testInitNormalizer) {
    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(3600);

    ml::api::CNdJsonOutputWriter outputWriter;

    ml::model::CLimits limits(false);
    ml::api::CResultNormalizer normalizer(modelConfig, outputWriter, limits);

    BOOST_TEST_REQUIRE(normalizer.initNormalizer("testfiles/quantilesState.json"));

    std::ifstream inputStrm("testfiles/normalizerInput.csv");
    ml::api::CCsvInputParser inputParser(inputStrm);
    BOOST_TEST_REQUIRE(inputParser.readStreamIntoMaps(std::bind(
        &ml::api::CResultNormalizer::handleRecord, &normalizer, std::placeholders::_1)));

    std::string results(outputWriter.internalString());
    LOG_DEBUG(<< "Results:\n" << results);

    // Results are new line separated so read all the docs into an  array
    std::vector<json::value> resultDocs;
    std::stringstream ss(results);
    std::string docString;
    json::error_code ec;
    while (std::getline(ss, docString)) {
        json::value doc = json::parse(docString, ec);
        BOOST_TEST_REQUIRE(ec.failed() == false);
        resultDocs.push_back(doc);
    }

    BOOST_REQUIRE_EQUAL(std::vector<json::value>::size_type{38}, resultDocs.size());

    // The maximum bucketTime influencer probability in the Savvis data used to initialise
    // the normaliser is 2.56098e-205, so this should map to the highest normalised
    // score which is 98.28496
    {
        const json::value& doc = resultDocs[0];
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("2.56098e-205"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("bucketTime"),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("root"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("98.28496"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[1];
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("2.93761e-203"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("status"),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("inflb"),
                            std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("97.26764"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[2];
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("5.56572e-204"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("status"),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("infl"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("98.56057"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[4];
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("count"),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("1e-300"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("status"),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("leaf"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("99.19481"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[15];
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("1e-10"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("bucketTime"),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("root"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("53.72631"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[35];
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("1"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("bucketTime"),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("root"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("0"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[36];
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("1"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("status"),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("infl"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("0"),
                            std::string(doc.at("normalized_score").as_string()));
    }
    {
        const json::value& doc = resultDocs[37];
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("value_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("count"),
                            std::string(doc.at("function_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("1"),
                            std::string(doc.at("probability").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("status"),
                            std::string(doc.at("person_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string(""),
                            std::string(doc.at("partition_field_name").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("leaf"), std::string(doc.at("level").as_string()));
        BOOST_REQUIRE_EQUAL(std::string("0"),
                            std::string(doc.at("normalized_score").as_string()));
    }
}

BOOST_AUTO_TEST_SUITE_END()
