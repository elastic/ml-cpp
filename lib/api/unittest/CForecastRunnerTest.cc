/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CForecastRunnerTest.h"

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/Constants.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <api/CAnomalyJob.h>
#include <api/CFieldConfig.h>

#include <rapidjson/document.h>

#include <cmath>
#include <memory>
#include <string>

namespace {

using TGenerateRecord = void (*)(ml::core_t::TTime time, ml::api::CAnomalyJob::TStrStrUMap& dataRows);

const ml::core_t::TTime START_TIME{12000000};
const ml::core_t::TTime BUCKET_LENGTH{3600};

void generateRecord(ml::core_t::TTime time, ml::api::CAnomalyJob::TStrStrUMap& dataRows) {
    dataRows["time"] = ml::core::CStringUtils::typeToString(time);
}

void generateRecordWithSummaryCount(ml::core_t::TTime time, ml::api::CAnomalyJob::TStrStrUMap& dataRows) {
    double x = static_cast<double>(time - START_TIME) / BUCKET_LENGTH;
    double count = (std::sin(x / 4.0) + 1.0) * 42.0 * std::pow(1.005, x);
    dataRows["time"] = ml::core::CStringUtils::typeToString(time);
    dataRows["count"] = ml::core::CStringUtils::typeToString(count);
}

void generateRecordWithStatus(ml::core_t::TTime time, ml::api::CAnomalyJob::TStrStrUMap& dataRows) {
    dataRows["time"] = ml::core::CStringUtils::typeToString(time);
    dataRows["status"] = (time / BUCKET_LENGTH) % 919 == 0 ? "404" : "200";
}

void generatePopulationRecord(ml::core_t::TTime time, ml::api::CAnomalyJob::TStrStrUMap& dataRows) {
    dataRows["time"] = ml::core::CStringUtils::typeToString(time);
    dataRows["person"] = "jill";
}

void populateJob(TGenerateRecord generateRecord, ml::api::CAnomalyJob& job, std::size_t buckets = 1000) {
    ml::core_t::TTime time = START_TIME;
    ml::api::CAnomalyJob::TStrStrUMap dataRows;
    for (std::size_t bucket = 0u; bucket < 2 * buckets; ++bucket, time += (BUCKET_LENGTH / 2)) {
        generateRecord(time, dataRows);
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
    }

    CPPUNIT_ASSERT_EQUAL(uint64_t(2 * buckets), job.numRecordsHandled());
}
}

void CForecastRunnerTest::testSummaryCount() {
    LOG_INFO(<< "*** test forecast on summary count ***");

    std::stringstream outputStrm;
    {
        ml::core::CJsonOutputStreamWrapper streamWrapper(outputStrm);
        ml::model::CLimits limits;
        ml::api::CFieldConfig fieldConfig;
        ml::api::CFieldConfig::TStrVec clauses;
        clauses.push_back("count");
        clauses.push_back("summarycountfield=count");
        fieldConfig.initFromClause(clauses);
        ml::model::CAnomalyDetectorModelConfig modelConfig = ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_LENGTH);

        ml::api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, streamWrapper);
        populateJob(generateRecordWithSummaryCount, job);

        ml::api::CAnomalyJob::TStrStrUMap dataRows;
        dataRows["."] = "p{\"duration\":" + std::to_string(13 * BUCKET_LENGTH) + ",\"forecast_id\": \"42\"" +
                        ",\"forecast_alias\": \"sumcount\"" + ",\"create_time\": \"1511370819\"" + ",\"expires_in\": \"" +
                        std::to_string(100 * ml::core::constants::DAY) + "\" }";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
    }

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(outputStrm.str());
    CPPUNIT_ASSERT(!doc.HasParseError());
    CPPUNIT_ASSERT(doc.GetArray().Size() > 0);
    bool foundScheduledRecord = false;
    bool foundStartedRecord = false;
    for (const auto& m : doc.GetArray()) {
        if (m.HasMember("model_forecast_request_stats")) {
            const rapidjson::Value& forecastStart = m["model_forecast_request_stats"];
            if (std::strcmp("scheduled", forecastStart["forecast_status"].GetString()) == 0) {
                CPPUNIT_ASSERT(!foundStartedRecord);
                foundScheduledRecord = true;
            } else if (std::strcmp("started", forecastStart["forecast_status"].GetString()) == 0) {
                CPPUNIT_ASSERT(foundScheduledRecord);
                foundStartedRecord = true;
                break;
            }
        }
    }
    CPPUNIT_ASSERT(foundScheduledRecord);
    CPPUNIT_ASSERT(foundStartedRecord);

    const rapidjson::Value& lastElement = doc[doc.GetArray().Size() - 1];
    CPPUNIT_ASSERT(lastElement.HasMember("model_forecast_request_stats"));
    const rapidjson::Value& forecastStats = lastElement["model_forecast_request_stats"];

    CPPUNIT_ASSERT_EQUAL(std::string("42"), std::string(forecastStats["forecast_id"].GetString()));
    CPPUNIT_ASSERT_EQUAL(std::string("sumcount"), std::string(forecastStats["forecast_alias"].GetString()));
    CPPUNIT_ASSERT_EQUAL(1511370819 * int64_t(1000), forecastStats["forecast_create_timestamp"].GetInt64());
    CPPUNIT_ASSERT(forecastStats.HasMember("processed_record_count"));
    CPPUNIT_ASSERT_EQUAL(13, forecastStats["processed_record_count"].GetInt());
    CPPUNIT_ASSERT_EQUAL(1.0, forecastStats["forecast_progress"].GetDouble());
    CPPUNIT_ASSERT_EQUAL(std::string("finished"), std::string(forecastStats["forecast_status"].GetString()));
    CPPUNIT_ASSERT_EQUAL(15591600 * int64_t(1000), forecastStats["timestamp"].GetInt64());
    CPPUNIT_ASSERT_EQUAL(15591600 * int64_t(1000), forecastStats["forecast_start_timestamp"].GetInt64());
    CPPUNIT_ASSERT_EQUAL((15591600 + 13 * BUCKET_LENGTH) * int64_t(1000), forecastStats["forecast_end_timestamp"].GetInt64());
    CPPUNIT_ASSERT_EQUAL((1511370819 + 100 * ml::core::constants::DAY) * int64_t(1000),
                         forecastStats["forecast_expiry_timestamp"].GetInt64());
}

void CForecastRunnerTest::testPopulation() {
    LOG_INFO(<< "*** test forecast on population ***");

    std::stringstream outputStrm;
    {
        ml::core::CJsonOutputStreamWrapper streamWrapper(outputStrm);
        ml::model::CLimits limits;
        ml::api::CFieldConfig fieldConfig;
        ml::api::CFieldConfig::TStrVec clauses;
        clauses.push_back("count");
        clauses.push_back("over");
        clauses.push_back("person");
        fieldConfig.initFromClause(clauses);
        ml::model::CAnomalyDetectorModelConfig modelConfig = ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_LENGTH);

        ml::api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, streamWrapper);
        populateJob(generatePopulationRecord, job);

        ml::api::CAnomalyJob::TStrStrUMap dataRows;
        dataRows["."] =
            "p{\"duration\":" + std::to_string(13 * BUCKET_LENGTH) + ",\"forecast_id\": \"31\"" + ",\"create_time\": \"1511370819\" }";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
    }

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(outputStrm.str());
    CPPUNIT_ASSERT(!doc.HasParseError());
    const rapidjson::Value& lastElement = doc[doc.GetArray().Size() - 1];
    CPPUNIT_ASSERT(lastElement.HasMember("model_forecast_request_stats"));
    const rapidjson::Value& forecastStats = lastElement["model_forecast_request_stats"];

    CPPUNIT_ASSERT(!doc.HasParseError());
    CPPUNIT_ASSERT_EQUAL(std::string("31"), std::string(forecastStats["forecast_id"].GetString()));
    CPPUNIT_ASSERT(!forecastStats.HasMember("forecast_alias"));
    CPPUNIT_ASSERT_EQUAL(std::string("failed"), std::string(forecastStats["forecast_status"].GetString()));
    CPPUNIT_ASSERT_EQUAL(ml::api::CForecastRunner::ERROR_NOT_SUPPORTED_FOR_POPULATION_MODELS,
                         std::string(forecastStats["forecast_messages"].GetArray()[0].GetString()));
    CPPUNIT_ASSERT_EQUAL((1511370819 + 14 * ml::core::constants::DAY) * int64_t(1000),
                         forecastStats["forecast_expiry_timestamp"].GetInt64());
}

void CForecastRunnerTest::testRare() {
    LOG_INFO(<< "*** test forecast on rare ***");

    std::stringstream outputStrm;
    {
        ml::core::CJsonOutputStreamWrapper streamWrapper(outputStrm);
        ml::model::CLimits limits;
        ml::api::CFieldConfig fieldConfig;
        ml::api::CFieldConfig::TStrVec clauses;
        clauses.push_back("rare");
        clauses.push_back("by");
        clauses.push_back("status");

        fieldConfig.initFromClause(clauses);
        ml::model::CAnomalyDetectorModelConfig modelConfig = ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_LENGTH);

        ml::api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, streamWrapper);
        populateJob(generateRecordWithStatus, job, 5000);

        ml::api::CAnomalyJob::TStrStrUMap dataRows;
        dataRows["."] = "p{\"duration\":" + std::to_string(13 * BUCKET_LENGTH) + ",\"forecast_id\": \"42\"" +
                        ",\"create_time\": \"1511370819\"" + ",\"expires_in\": \"8640000\" }";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
    }
    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(outputStrm.str());
    CPPUNIT_ASSERT(!doc.HasParseError());
    const rapidjson::Value& lastElement = doc[doc.GetArray().Size() - 1];
    CPPUNIT_ASSERT(lastElement.HasMember("model_forecast_request_stats"));
    const rapidjson::Value& forecastStats = lastElement["model_forecast_request_stats"];

    CPPUNIT_ASSERT(!doc.HasParseError());
    CPPUNIT_ASSERT_EQUAL(std::string("42"), std::string(forecastStats["forecast_id"].GetString()));
    CPPUNIT_ASSERT(!forecastStats.HasMember("forecast_alias"));
    CPPUNIT_ASSERT_EQUAL(std::string("failed"), std::string(forecastStats["forecast_status"].GetString()));
    CPPUNIT_ASSERT_EQUAL(ml::api::CForecastRunner::ERROR_NO_SUPPORTED_FUNCTIONS,
                         std::string(forecastStats["forecast_messages"].GetArray()[0].GetString()));
    CPPUNIT_ASSERT_EQUAL((1511370819 + 14 * ml::core::constants::DAY) * int64_t(1000),
                         forecastStats["forecast_expiry_timestamp"].GetInt64());
}

void CForecastRunnerTest::testInsufficientData() {
    LOG_INFO(<< "*** test insufficient data ***");

    std::stringstream outputStrm;
    {
        ml::core::CJsonOutputStreamWrapper streamWrapper(outputStrm);
        ml::model::CLimits limits;
        ml::api::CFieldConfig fieldConfig;
        ml::api::CFieldConfig::TStrVec clauses;
        clauses.push_back("count");
        fieldConfig.initFromClause(clauses);
        ml::model::CAnomalyDetectorModelConfig modelConfig = ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_LENGTH);

        ml::api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, streamWrapper);
        populateJob(generateRecord, job, 3);

        ml::api::CAnomalyJob::TStrStrUMap dataRows;
        dataRows["."] =
            "p{\"duration\":" + std::to_string(13 * BUCKET_LENGTH) + ",\"forecast_id\": \"31\"" + ",\"create_time\": \"1511370819\" }";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
    }

    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(outputStrm.str());
    CPPUNIT_ASSERT(!doc.HasParseError());
    const rapidjson::Value& lastElement = doc[doc.GetArray().Size() - 1];
    CPPUNIT_ASSERT(lastElement.HasMember("model_forecast_request_stats"));
    const rapidjson::Value& forecastStats = lastElement["model_forecast_request_stats"];

    CPPUNIT_ASSERT(!doc.HasParseError());
    CPPUNIT_ASSERT_EQUAL(std::string("31"), std::string(forecastStats["forecast_id"].GetString()));
    CPPUNIT_ASSERT_EQUAL(std::string("finished"), std::string(forecastStats["forecast_status"].GetString()));
    CPPUNIT_ASSERT_EQUAL(1.0, forecastStats["forecast_progress"].GetDouble());
    CPPUNIT_ASSERT_EQUAL(ml::api::CForecastRunner::INFO_NO_MODELS_CAN_CURRENTLY_BE_FORECAST,
                         std::string(forecastStats["forecast_messages"].GetArray()[0].GetString()));
    CPPUNIT_ASSERT_EQUAL((1511370819 + 14 * ml::core::constants::DAY) * int64_t(1000),
                         forecastStats["forecast_expiry_timestamp"].GetInt64());
}

void CForecastRunnerTest::testValidateDuration() {
    ml::api::CForecastRunner::SForecast forecastJob;

    std::string message("p{\"duration\":" + std::to_string(10 * ml::core::constants::WEEK) + ",\"forecast_id\": \"42\"" +
                        ",\"create_time\": \"1511370819\" }");

    CPPUNIT_ASSERT(ml::api::CForecastRunner::parseAndValidateForecastRequest(message, forecastJob, 1400000000));
    CPPUNIT_ASSERT_EQUAL(8 * ml::core::constants::WEEK, forecastJob.s_Duration);
    CPPUNIT_ASSERT_EQUAL(8 * ml::core::constants::WEEK + 1400000000, forecastJob.forecastEnd());
    CPPUNIT_ASSERT_EQUAL(ml::api::CForecastRunner::WARNING_DURATION_LIMIT, *forecastJob.s_Messages.begin());
}

void CForecastRunnerTest::testValidateDefaultExpiry() {
    ml::api::CForecastRunner::SForecast forecastJob;

    std::string message("p{\"duration\":" + std::to_string(2 * ml::core::constants::WEEK) + ",\"forecast_id\": \"42\"" +
                        ",\"create_time\": \"1511370819\" }");

    CPPUNIT_ASSERT(ml::api::CForecastRunner::parseAndValidateForecastRequest(message, forecastJob, 1400000000));
    CPPUNIT_ASSERT_EQUAL(2 * ml::core::constants::WEEK, forecastJob.s_Duration);
    CPPUNIT_ASSERT_EQUAL(14 * ml::core::constants::DAY + 1511370819, forecastJob.s_ExpiryTime);

    std::string message2("p{\"duration\":" + std::to_string(2 * ml::core::constants::WEEK) + ",\"forecast_id\": \"42\"" +
                         ",\"create_time\": \"1511370819\"" + ",\"expires_in\": -1 }");
    CPPUNIT_ASSERT(ml::api::CForecastRunner::parseAndValidateForecastRequest(message2, forecastJob, 1400000000));
    CPPUNIT_ASSERT_EQUAL(2 * ml::core::constants::WEEK, forecastJob.s_Duration);
    CPPUNIT_ASSERT_EQUAL(14 * ml::core::constants::DAY + 1511370819, forecastJob.s_ExpiryTime);
}

void CForecastRunnerTest::testValidateNoExpiry() {
    ml::api::CForecastRunner::SForecast forecastJob;

    std::string message("p{\"duration\":" + std::to_string(3 * ml::core::constants::WEEK) + ",\"forecast_id\": \"42\"" +
                        ",\"create_time\": \"1511370819\"" + ",\"expires_in\": 0 }");

    CPPUNIT_ASSERT(ml::api::CForecastRunner::parseAndValidateForecastRequest(message, forecastJob, 1400000000));
    CPPUNIT_ASSERT_EQUAL(3 * ml::core::constants::WEEK, forecastJob.s_Duration);
    CPPUNIT_ASSERT_EQUAL(ml::core_t::TTime(1511370819), forecastJob.s_ExpiryTime);
    CPPUNIT_ASSERT_EQUAL(forecastJob.s_CreateTime, forecastJob.s_ExpiryTime);
}

void CForecastRunnerTest::testValidateInvalidExpiry() {
    ml::api::CForecastRunner::SForecast forecastJob;

    std::string message("p{\"duration\":" + std::to_string(3 * ml::core::constants::WEEK) + ",\"forecast_id\": \"42\"" +
                        ",\"create_time\": \"1511370819\"" + ",\"expires_in\": -244 }");

    CPPUNIT_ASSERT(ml::api::CForecastRunner::parseAndValidateForecastRequest(message, forecastJob, 1400000000));
    CPPUNIT_ASSERT_EQUAL(3 * ml::core::constants::WEEK, forecastJob.s_Duration);
    CPPUNIT_ASSERT_EQUAL(14 * ml::core::constants::DAY + 1511370819, forecastJob.s_ExpiryTime);
}

void CForecastRunnerTest::testValidateBrokenMessage() {
    ml::api::CForecastRunner::SForecast forecastJob;

    std::string message("p{\"dura");

    CPPUNIT_ASSERT(ml::api::CForecastRunner::parseAndValidateForecastRequest(message, forecastJob, 1400000000) == false);
}

void CForecastRunnerTest::testValidateMissingId() {
    ml::api::CForecastRunner::SForecast forecastJob;

    std::string message("p{\"duration\":" + std::to_string(3 * ml::core::constants::WEEK) + ",\"create_time\": \"1511370819\"}");

    CPPUNIT_ASSERT(ml::api::CForecastRunner::parseAndValidateForecastRequest(message, forecastJob, 1400000000) == false);
}

CppUnit::Test* CForecastRunnerTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CForecastRunnerTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<CForecastRunnerTest>("CForecastRunnerTest::testSummaryCount", &CForecastRunnerTest::testSummaryCount));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CForecastRunnerTest>("CForecastRunnerTest::testPopulation", &CForecastRunnerTest::testPopulation));
    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastRunnerTest>("CForecastRunnerTest::testRare", &CForecastRunnerTest::testRare));
    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastRunnerTest>("CForecastRunnerTest::testInsufficientData",
                                                                       &CForecastRunnerTest::testInsufficientData));
    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastRunnerTest>("CForecastRunnerTest::testValidateDuration",
                                                                       &CForecastRunnerTest::testValidateDuration));
    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastRunnerTest>("CForecastRunnerTest::testValidateExpiry",
                                                                       &CForecastRunnerTest::testValidateDefaultExpiry));
    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastRunnerTest>("CForecastRunnerTest::testValidateNoExpiry",
                                                                       &CForecastRunnerTest::testValidateNoExpiry));
    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastRunnerTest>("CForecastRunnerTest::testValidateInvalidExpiry",
                                                                       &CForecastRunnerTest::testValidateInvalidExpiry));
    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastRunnerTest>("CForecastRunnerTest::testBrokenMessage",
                                                                       &CForecastRunnerTest::testValidateBrokenMessage));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CForecastRunnerTest>("CForecastRunnerTest::testMissingId", &CForecastRunnerTest::testValidateMissingId));

    return suiteOfTests;
}
