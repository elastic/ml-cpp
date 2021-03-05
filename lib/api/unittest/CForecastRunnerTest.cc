/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/Constants.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <api/CAnomalyJobConfig.h>

#include "CTestAnomalyJob.h"

#include <rapidjson/document.h>

#include <boost/test/unit_test.hpp>

#include <cmath>
#include <memory>
#include <string>

BOOST_AUTO_TEST_SUITE(CForecastRunnerTest)

namespace {

using TGenerateRecord = void (*)(ml::core_t::TTime time,
                                 CTestAnomalyJob::TStrStrUMap& dataRows);

const ml::core_t::TTime START_TIME{12000000};
const ml::core_t::TTime BUCKET_LENGTH{3600};

void generateRecord(ml::core_t::TTime time, CTestAnomalyJob::TStrStrUMap& dataRows) {
    dataRows["time"] = ml::core::CStringUtils::typeToString(time);
}

void generateRecordWithSummaryCount(ml::core_t::TTime time,
                                    CTestAnomalyJob::TStrStrUMap& dataRows) {
    double x = static_cast<double>(time - START_TIME) / BUCKET_LENGTH;
    double count = (std::sin(x / 4.0) + 1.0) * 42.0 * std::pow(1.005, x);
    dataRows["time"] = ml::core::CStringUtils::typeToString(time);
    dataRows["count"] = ml::core::CStringUtils::typeToString(count);
}

void generateRecordWithStatus(ml::core_t::TTime time, CTestAnomalyJob::TStrStrUMap& dataRows) {
    dataRows["time"] = ml::core::CStringUtils::typeToString(time);
    dataRows["status"] = (time / BUCKET_LENGTH) % 919 == 0 ? "404" : "200";
}

void generatePopulationRecord(ml::core_t::TTime time, CTestAnomalyJob::TStrStrUMap& dataRows) {
    dataRows["time"] = ml::core::CStringUtils::typeToString(time);
    dataRows["person"] = "jill";
}

void populateJob(TGenerateRecord generateRecord, CTestAnomalyJob& job, std::size_t buckets = 1000) {
    ml::core_t::TTime time = START_TIME;
    CTestAnomalyJob::TStrStrUMap dataRows;
    for (std::size_t bucket = 0; bucket < 2 * buckets;
         ++bucket, time += (BUCKET_LENGTH / 2)) {
        generateRecord(time, dataRows);
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
    }

    BOOST_REQUIRE_EQUAL(uint64_t(2 * buckets), job.numRecordsHandled());
}
}

BOOST_AUTO_TEST_CASE(testSummaryCount) {
    std::stringstream outputStrm;
    {
        ml::core::CJsonOutputStreamWrapper streamWrapper(outputStrm);
        ml::model::CLimits limits;
        ml::api::CAnomalyJobConfig jobConfig =
            CTestAnomalyJob::makeSimpleJobConfig("count", "", "", "", "", {}, "count");

        ml::model::CAnomalyDetectorModelConfig modelConfig =
            ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_LENGTH);

        CTestAnomalyJob job("job", limits, jobConfig, modelConfig, streamWrapper);
        populateJob(generateRecordWithSummaryCount, job);

        CTestAnomalyJob::TStrStrUMap dataRows;
        dataRows["."] = "p{\"duration\":" + std::to_string(13 * BUCKET_LENGTH) +
                        ",\"forecast_id\": \"42\"" + ",\"forecast_alias\": \"sumcount\"" +
                        ",\"create_time\": \"1511370819\"" + ",\"expires_in\": \"" +
                        std::to_string(100 * ml::core::constants::DAY) + "\" }";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
    }

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(outputStrm.str());
    BOOST_TEST_REQUIRE(!doc.HasParseError());
    BOOST_TEST_REQUIRE(doc.GetArray().Size() > 0);
    bool foundScheduledRecord = false;
    bool foundStartedRecord = false;
    for (const auto& m : doc.GetArray()) {
        if (m.HasMember("model_forecast_request_stats")) {
            const rapidjson::Value& forecastStart = m["model_forecast_request_stats"];
            if (std::strcmp("scheduled", forecastStart["forecast_status"].GetString()) == 0) {
                BOOST_TEST_REQUIRE(!foundStartedRecord);
                foundScheduledRecord = true;
            } else if (std::strcmp("started",
                                   forecastStart["forecast_status"].GetString()) == 0) {
                BOOST_TEST_REQUIRE(foundScheduledRecord);
                foundStartedRecord = true;
                break;
            }
        }
    }
    BOOST_TEST_REQUIRE(foundScheduledRecord);
    BOOST_TEST_REQUIRE(foundStartedRecord);

    const rapidjson::Value& lastElement = doc[doc.GetArray().Size() - 1];
    BOOST_TEST_REQUIRE(lastElement.HasMember("model_forecast_request_stats"));
    const rapidjson::Value& forecastStats = lastElement["model_forecast_request_stats"];

    BOOST_REQUIRE_EQUAL(std::string("42"),
                        std::string(forecastStats["forecast_id"].GetString()));
    BOOST_REQUIRE_EQUAL(std::string("sumcount"),
                        std::string(forecastStats["forecast_alias"].GetString()));
    BOOST_REQUIRE_EQUAL(1511370819 * int64_t(1000),
                        forecastStats["forecast_create_timestamp"].GetInt64());
    BOOST_TEST_REQUIRE(forecastStats.HasMember("processed_record_count"));
    BOOST_REQUIRE_EQUAL(13, forecastStats["processed_record_count"].GetInt());
    BOOST_REQUIRE_EQUAL(1.0, forecastStats["forecast_progress"].GetDouble());
    BOOST_REQUIRE_EQUAL(std::string("finished"),
                        std::string(forecastStats["forecast_status"].GetString()));
    BOOST_REQUIRE_EQUAL(15591600 * int64_t(1000), forecastStats["timestamp"].GetInt64());
    BOOST_REQUIRE_EQUAL(15591600 * int64_t(1000),
                        forecastStats["forecast_start_timestamp"].GetInt64());
    BOOST_REQUIRE_EQUAL((15591600 + 13 * BUCKET_LENGTH) * int64_t(1000),
                        forecastStats["forecast_end_timestamp"].GetInt64());
    BOOST_REQUIRE_EQUAL((1511370819 + 100 * ml::core::constants::DAY) * int64_t(1000),
                        forecastStats["forecast_expiry_timestamp"].GetInt64());
}

BOOST_AUTO_TEST_CASE(testPopulation) {
    std::stringstream outputStrm;
    {
        ml::core::CJsonOutputStreamWrapper streamWrapper(outputStrm);
        ml::model::CLimits limits;
        ml::api::CAnomalyJobConfig jobConfig =
            CTestAnomalyJob::makeSimpleJobConfig("count", "", "", "person", "");

        ml::model::CAnomalyDetectorModelConfig modelConfig =
            ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_LENGTH);

        CTestAnomalyJob job("job", limits, jobConfig, modelConfig, streamWrapper);
        populateJob(generatePopulationRecord, job);

        CTestAnomalyJob::TStrStrUMap dataRows;
        dataRows["."] = "p{\"duration\":" + std::to_string(13 * BUCKET_LENGTH) +
                        ",\"forecast_id\": \"31\"" + ",\"create_time\": \"1511370819\" }";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
    }

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(outputStrm.str());
    BOOST_TEST_REQUIRE(!doc.HasParseError());
    const rapidjson::Value& lastElement = doc[doc.GetArray().Size() - 1];
    BOOST_TEST_REQUIRE(lastElement.HasMember("model_forecast_request_stats"));
    const rapidjson::Value& forecastStats = lastElement["model_forecast_request_stats"];

    BOOST_TEST_REQUIRE(!doc.HasParseError());
    BOOST_REQUIRE_EQUAL(std::string("31"),
                        std::string(forecastStats["forecast_id"].GetString()));
    BOOST_TEST_REQUIRE(!forecastStats.HasMember("forecast_alias"));
    BOOST_REQUIRE_EQUAL(std::string("failed"),
                        std::string(forecastStats["forecast_status"].GetString()));
    BOOST_REQUIRE_EQUAL(
        ml::api::CForecastRunner::ERROR_NOT_SUPPORTED_FOR_POPULATION_MODELS,
        std::string(forecastStats["forecast_messages"].GetArray()[0].GetString()));
    BOOST_REQUIRE_EQUAL((1511370819 + 14 * ml::core::constants::DAY) * int64_t(1000),
                        forecastStats["forecast_expiry_timestamp"].GetInt64());
}

BOOST_AUTO_TEST_CASE(testRare) {
    std::stringstream outputStrm;
    {
        ml::core::CJsonOutputStreamWrapper streamWrapper(outputStrm);
        ml::model::CLimits limits;
        ml::api::CAnomalyJobConfig jobConfig =
            CTestAnomalyJob::makeSimpleJobConfig("rare", "", "status", "", "");

        ml::model::CAnomalyDetectorModelConfig modelConfig =
            ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_LENGTH);

        CTestAnomalyJob job("job", limits, jobConfig, modelConfig, streamWrapper);
        populateJob(generateRecordWithStatus, job, 5000);

        CTestAnomalyJob::TStrStrUMap dataRows;
        dataRows["."] = "p{\"duration\":" + std::to_string(13 * BUCKET_LENGTH) +
                        ",\"forecast_id\": \"42\"" + ",\"create_time\": \"1511370819\"" +
                        ",\"expires_in\": \"8640000\" }";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
    }
    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(outputStrm.str());
    BOOST_TEST_REQUIRE(!doc.HasParseError());
    const rapidjson::Value& lastElement = doc[doc.GetArray().Size() - 1];
    BOOST_TEST_REQUIRE(lastElement.HasMember("model_forecast_request_stats"));
    const rapidjson::Value& forecastStats = lastElement["model_forecast_request_stats"];

    BOOST_TEST_REQUIRE(!doc.HasParseError());
    BOOST_REQUIRE_EQUAL(std::string("42"),
                        std::string(forecastStats["forecast_id"].GetString()));
    BOOST_TEST_REQUIRE(!forecastStats.HasMember("forecast_alias"));
    BOOST_REQUIRE_EQUAL(std::string("failed"),
                        std::string(forecastStats["forecast_status"].GetString()));
    BOOST_REQUIRE_EQUAL(
        ml::api::CForecastRunner::ERROR_NO_SUPPORTED_FUNCTIONS,
        std::string(forecastStats["forecast_messages"].GetArray()[0].GetString()));
    BOOST_REQUIRE_EQUAL((1511370819 + 14 * ml::core::constants::DAY) * int64_t(1000),
                        forecastStats["forecast_expiry_timestamp"].GetInt64());
}

BOOST_AUTO_TEST_CASE(testInsufficientData) {
    std::stringstream outputStrm;
    {
        ml::core::CJsonOutputStreamWrapper streamWrapper(outputStrm);
        ml::model::CLimits limits;
        ml::api::CAnomalyJobConfig jobConfig =
            CTestAnomalyJob::makeSimpleJobConfig("count", "", "", "", "");

        ml::model::CAnomalyDetectorModelConfig modelConfig =
            ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_LENGTH);

        CTestAnomalyJob job("job", limits, jobConfig, modelConfig, streamWrapper);
        populateJob(generateRecord, job, 3);

        CTestAnomalyJob::TStrStrUMap dataRows;
        dataRows["."] = "p{\"duration\":" + std::to_string(13 * BUCKET_LENGTH) +
                        ",\"forecast_id\": \"31\"" + ",\"create_time\": \"1511370819\" }";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
    }

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(outputStrm.str());
    BOOST_TEST_REQUIRE(!doc.HasParseError());
    const rapidjson::Value& lastElement = doc[doc.GetArray().Size() - 1];
    BOOST_TEST_REQUIRE(lastElement.HasMember("model_forecast_request_stats"));
    const rapidjson::Value& forecastStats = lastElement["model_forecast_request_stats"];

    BOOST_TEST_REQUIRE(!doc.HasParseError());
    BOOST_REQUIRE_EQUAL(std::string("31"),
                        std::string(forecastStats["forecast_id"].GetString()));
    BOOST_REQUIRE_EQUAL(std::string("finished"),
                        std::string(forecastStats["forecast_status"].GetString()));
    BOOST_REQUIRE_EQUAL(1.0, forecastStats["forecast_progress"].GetDouble());
    BOOST_REQUIRE_EQUAL(
        ml::api::CForecastRunner::INFO_NO_MODELS_CAN_CURRENTLY_BE_FORECAST,
        std::string(forecastStats["forecast_messages"].GetArray()[0].GetString()));
    BOOST_REQUIRE_EQUAL((1511370819 + 14 * ml::core::constants::DAY) * int64_t(1000),
                        forecastStats["forecast_expiry_timestamp"].GetInt64());
}

BOOST_AUTO_TEST_CASE(testValidateDefaultExpiry) {
    ml::api::CForecastRunner::SForecast forecastJob;

    std::string message("p{\"duration\":" + std::to_string(2 * ml::core::constants::WEEK) +
                        ",\"forecast_id\": \"42\"" + ",\"create_time\": \"1511370819\" }");

    BOOST_TEST_REQUIRE(ml::api::CForecastRunner::parseAndValidateForecastRequest(
        message, forecastJob, 1400000000));
    BOOST_REQUIRE_EQUAL(2 * ml::core::constants::WEEK, forecastJob.s_Duration);
    BOOST_REQUIRE_EQUAL(14 * ml::core::constants::DAY + 1511370819, forecastJob.s_ExpiryTime);

    std::string message2("p{\"duration\":" + std::to_string(2 * ml::core::constants::WEEK) +
                         ",\"forecast_id\": \"42\"" +
                         ",\"create_time\": \"1511370819\"" + ",\"expires_in\": -1 }");
    BOOST_TEST_REQUIRE(ml::api::CForecastRunner::parseAndValidateForecastRequest(
        message2, forecastJob, 1400000000));
    BOOST_REQUIRE_EQUAL(2 * ml::core::constants::WEEK, forecastJob.s_Duration);
    BOOST_REQUIRE_EQUAL(14 * ml::core::constants::DAY + 1511370819, forecastJob.s_ExpiryTime);
}

BOOST_AUTO_TEST_CASE(testValidateNoExpiry) {
    ml::api::CForecastRunner::SForecast forecastJob;

    std::string message("p{\"duration\":" + std::to_string(3 * ml::core::constants::WEEK) +
                        ",\"forecast_id\": \"42\"" +
                        ",\"create_time\": \"1511370819\"" + ",\"expires_in\": 0 }");

    BOOST_TEST_REQUIRE(ml::api::CForecastRunner::parseAndValidateForecastRequest(
        message, forecastJob, 1400000000));
    BOOST_REQUIRE_EQUAL(3 * ml::core::constants::WEEK, forecastJob.s_Duration);
    BOOST_REQUIRE_EQUAL(ml::core_t::TTime(1511370819), forecastJob.s_ExpiryTime);
    BOOST_REQUIRE_EQUAL(forecastJob.s_CreateTime, forecastJob.s_ExpiryTime);
}

BOOST_AUTO_TEST_CASE(testValidateInvalidExpiry) {
    ml::api::CForecastRunner::SForecast forecastJob;

    std::string message("p{\"duration\":" + std::to_string(3 * ml::core::constants::WEEK) +
                        ",\"forecast_id\": \"42\"" + ",\"create_time\": \"1511370819\"" +
                        ",\"expires_in\": -244 }");

    BOOST_TEST_REQUIRE(ml::api::CForecastRunner::parseAndValidateForecastRequest(
        message, forecastJob, 1400000000));
    BOOST_REQUIRE_EQUAL(3 * ml::core::constants::WEEK, forecastJob.s_Duration);
    BOOST_REQUIRE_EQUAL(14 * ml::core::constants::DAY + 1511370819, forecastJob.s_ExpiryTime);
}

BOOST_AUTO_TEST_CASE(testValidateBrokenMessage) {
    ml::api::CForecastRunner::SForecast forecastJob;

    std::string message("p{\"dura");

    BOOST_TEST_REQUIRE(ml::api::CForecastRunner::parseAndValidateForecastRequest(
                           message, forecastJob, 1400000000) == false);
}

BOOST_AUTO_TEST_CASE(testValidateMissingId) {
    ml::api::CForecastRunner::SForecast forecastJob;

    std::string message("p{\"duration\":" + std::to_string(3 * ml::core::constants::WEEK) +
                        ",\"create_time\": \"1511370819\"}");

    BOOST_TEST_REQUIRE(ml::api::CForecastRunner::parseAndValidateForecastRequest(
                           message, forecastJob, 1400000000) == false);
}

BOOST_AUTO_TEST_CASE(testValidateProvidedMinDiskSpace) {
    ml::api::CForecastRunner::SForecast forecastJob;

    std::string message{
        "p{\"duration\":" + std::to_string(3 * ml::core::constants::WEEK) +
        ",\"forecast_id\": \"42\",\"create_time\": \"1511370819\",\"min_available_disk_space\": 100000}"};

    BOOST_TEST_REQUIRE(ml::api::CForecastRunner::parseAndValidateForecastRequest(
        message, forecastJob, 1400000000));
    BOOST_REQUIRE_EQUAL(100000, forecastJob.s_MinForecastAvailableDiskSpace);

    std::string message2{"p{\"duration\":" + std::to_string(3 * ml::core::constants::WEEK) +
                         ",\"forecast_id\": \"42\",\"create_time\": \"1511370819\"}"};

    BOOST_TEST_REQUIRE(ml::api::CForecastRunner::parseAndValidateForecastRequest(
        message2, forecastJob, 1400000000));
    BOOST_REQUIRE_EQUAL(ml::api::CForecastRunner::DEFAULT_MIN_FORECAST_AVAILABLE_DISK_SPACE,
                        forecastJob.s_MinForecastAvailableDiskSpace);
}

BOOST_AUTO_TEST_CASE(testValidateProvidedMaxMemoryLimit) {
    ml::api::CForecastRunner::SForecast forecastJob;

    std::string message{
        "p{\"duration\":" + std::to_string(3 * ml::core::constants::WEEK) +
        ",\"forecast_id\": \"42\",\"create_time\": \"1511370819\",\"max_model_memory\": 10000000}"};

    BOOST_TEST_REQUIRE(ml::api::CForecastRunner::parseAndValidateForecastRequest(
        message, forecastJob, 1400000000));
    BOOST_REQUIRE_EQUAL(10000000, forecastJob.s_MaxForecastModelMemory);

    std::string message2{"p{\"duration\":" + std::to_string(3 * ml::core::constants::WEEK) +
                         ",\"forecast_id\": \"42\",\"create_time\": \"1511370819\"}"};

    BOOST_TEST_REQUIRE(ml::api::CForecastRunner::parseAndValidateForecastRequest(
        message2, forecastJob, 1400000000));
    BOOST_REQUIRE_EQUAL(ml::api::CForecastRunner::DEFAULT_MAX_FORECAST_MODEL_MEMORY,
                        forecastJob.s_MaxForecastModelMemory);
}

BOOST_AUTO_TEST_CASE(testValidateProvidedTooLargeMaxMemoryLimit) {
    ml::api::CForecastRunner::SForecast forecastJob;

    std::string message("p{\"duration\":" + std::to_string(3 * ml::core::constants::WEEK) +
                        ",\"forecast_id\": \"42\",\"create_time\": \"1511370819\",\"max_model_memory\":" +
                        std::to_string(524288000ull + 10ull) + "}");

    // larger than the most we can persist to disk should cause a failure
    BOOST_TEST_REQUIRE(ml::api::CForecastRunner::parseAndValidateForecastRequest(
                           message, forecastJob, 1400000000,
                           std::numeric_limits<std::size_t>::max() / 2,
                           [](const ml::api::CForecastRunner::SForecast&,
                              const std::string&) { return; }) == false);

    std::string message2(
        "p{\"duration\":" + std::to_string(3 * ml::core::constants::WEEK) +
        ",\"forecast_id\": \"42\",\"create_time\": \"1511370819\",\"max_model_memory\":31457280}");

    // Larger than 40% of the configured job memory should fail
    BOOST_TEST_REQUIRE(ml::api::CForecastRunner::parseAndValidateForecastRequest(
                           message2, forecastJob, 1400000000, 31457280ull,
                           [](const ml::api::CForecastRunner::SForecast&,
                              const std::string&) { return; }) == false);

    // Less than 40% of the configured job memory should NOT fail
    BOOST_TEST_REQUIRE(ml::api::CForecastRunner::parseAndValidateForecastRequest(
        message2, forecastJob, 1400000000, static_cast<std::size_t>(31457280ull * 3),
        [](const ml::api::CForecastRunner::SForecast&, const std::string&) {
            return;
        }));
}

BOOST_AUTO_TEST_CASE(testSufficientDiskSpace) {

    // These tests could theoretically fail based on environmental factors, but
    // it's unlikely - they are saying the current directory must have at least
    // 1 byte free disk space and less than 16 exabytes free
    BOOST_REQUIRE_EQUAL(
        true, ml::api::CForecastRunner::sufficientAvailableDiskSpace(1, "."));
    BOOST_REQUIRE_EQUAL(false, ml::api::CForecastRunner::sufficientAvailableDiskSpace(
                                   std::numeric_limits<std::size_t>::max(), "."));
}

BOOST_AUTO_TEST_SUITE_END()
